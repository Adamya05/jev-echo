"""Four-way comparison of search configs on the Jev prior.

    uv run python -m sysone.snake.compare
    uv run python -m sysone.snake.compare --games 10 --iterations 320

UNIFORM+MCTS is the control. Read every other row against it: whatever
JEV+MCTS gains over UNIFORM+MCTS is attributable to the prior, and
whatever it gains over JEV alone is attributable to the search.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from sysone.core import get_backend
from sysone.snake import search
from sysone.snake.game import Snake

BOLD, DIM, RESET = "\x1b[1m", "\x1b[2m", "\x1b[0m"


@dataclass
class Result:
    config: str
    seed: int
    score: int = 0
    steps: int = 0
    calls: int = 0
    tokens: int = 0
    usd: float = 0.0
    wall_s: float = 0.0
    decide_ms: list[float] = field(default_factory=list)
    open_space: list[int] = field(default_factory=list)
    greedy_hits: int = 0
    decisions: int = 0
    death: str = "step limit"


def play(config: str, seed: int, iterations: int, max_steps: int,
         rollouts: int = 16) -> Result:
    needs_api = not (config in ("uniform_mcts", "laya", "laya_search",
                                "laya_raw", "laya_fast")
                     or config.startswith("board_")
                     or config.startswith("search_laya")
                     or config.startswith("student"))
    backend = (get_backend()
               if needs_api and config not in ("jev_or", "jev_search",
                                               "laya_search")
               and not config.startswith("search_")
               and config != "jev_raw"
               else None)
    pol = search.build(config, backend, iterations, rollouts)
    game = Snake(seed=seed)
    rng = random.Random(seed * 7919 + 13)
    r = Result(config=pol.name, seed=seed)

    t0 = time.perf_counter()
    while game.alive and game.steps < max_steps:
        facts = game.options()
        if not facts:
            r.death = "boxed in"
            break

        # Strategy-divergence telemetry, free from what analyse() already computes.
        r.open_space.append(int(statistics.mean(f.open_space for f in facts.values())))
        nearest = min(facts, key=lambda m: facts[m].food_distance)

        st = pol.select(game, rng)
        r.calls += st.calls
        r.tokens += st.tokens
        r.usd += st.usd
        if st.ms:
            r.decide_ms.append(st.ms)
        if len(facts) > 1:
            r.decisions += 1
            if st.move == nearest:
                r.greedy_hits += 1

        game.step(st.move)
        if not game.alive:
            r.death = "collision"

    r.wall_s = time.perf_counter() - t0
    r.score, r.steps = game.score, game.steps
    return r


def _job(args: tuple) -> Result:
    """Module-level so it survives pickling into a worker process."""
    return play(*args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=160)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--configs", nargs="*", default=search.CONFIGS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=16,
                    help="rollouts per candidate for best-of-N")
    ap.add_argument("--save", default=None, help="write per-seed results as JSON")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="first board; e.g. 100 for boards the tuning never saw")
    args = ap.parse_args()

    jobs = [(c, s) for c in args.configs
            for s in range(args.seed_offset, args.seed_offset + args.games)]
    print(f"\n{BOLD}{len(jobs)} games{RESET}  "
          f"{args.games} seeds x {len(args.configs)} configs, "
          f"{args.iterations} MCTS iterations, cap {args.max_steps} steps\n")

    t0 = time.perf_counter()
    payload = [(c, s, args.iterations, args.max_steps, args.rollouts)
               for c, s in jobs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_job, payload))
    wall = time.perf_counter() - t0

    by_config: dict[str, list[Result]] = {}
    for r in results:
        by_config.setdefault(r.config, []).append(r)

    hdr = (f"{'config':<22}{'score':>7}{'+/-':>7}{'steps':>7}{'calls':>7}{'ms/dec':>8}"
           f"{'$':>9}{'space':>7}{'greedy':>8}  deaths")
    print(BOLD + hdr + RESET)
    print(DIM + "-" * len(hdr) + RESET)

    for cfg in by_config:
        rs = by_config[cfg]
        scores = [r.score for r in rs]
        score = statistics.mean(scores)
        sem = (statistics.pstdev(scores) / len(scores) ** 0.5) if len(scores) > 1 else 0.0
        steps = statistics.mean(r.steps for r in rs)
        calls = statistics.mean(r.calls for r in rs)
        lat = [m for r in rs for m in r.decide_ms]
        usd = sum(r.usd for r in rs)
        space = statistics.mean(s for r in rs for s in r.open_space) if any(r.open_space for r in rs) else 0
        gh = sum(r.greedy_hits for r in rs)
        dec = sum(r.decisions for r in rs) or 1
        deaths = {}
        for r in rs:
            deaths[r.death] = deaths.get(r.death, 0) + 1
        dstr = ", ".join(f"{k} {v}" for k, v in sorted(deaths.items()))
        med = statistics.median(lat) if lat else 0.0
        ms = f"{med:.2f}" if 0 < med < 10 else f"{med:.0f}"
        print(f"{cfg:<22}{score:>7.1f}{sem:>7.2f}{steps:>7.0f}{calls:>7.0f}"
              f"{ms:>8}{usd:>9.4f}"
              f"{space:>7.0f}{gh / dec:>8.0%}  {dstr}")

    print(f"\n{DIM}wall clock {wall:.0f}s across {args.workers} workers   "
          f"total ${sum(r.usd for r in results):.4f}{RESET}")
    print(f"{DIM}score = mean food eaten (+/- standard error); space = mean open "
          f"cells per decision; greedy = % of decisions taking the food-nearest move{RESET}")

    # Paired comparison. Both configs played the same seeds, so differencing
    # per seed cancels seed variance -- far more power than an unpaired test.
    names = list(by_config)
    if len(names) > 1:
        print()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A = {r.seed: r.score for r in by_config[names[i]]}
                B = {r.seed: r.score for r in by_config[names[j]]}
                shared = sorted(set(A) & set(B))
                diffs = [A[s] - B[s] for s in shared]
                if len(diffs) < 2:
                    continue
                d = statistics.mean(diffs)
                se = statistics.stdev(diffs) / len(diffs) ** 0.5
                wins = sum(1 for x in diffs if x > 0)
                ties = sum(1 for x in diffs if x == 0)
                verdict = "distinguishable" if abs(d) > 2 * se else "NOT distinguishable"
                print(f"{BOLD}paired{RESET} {names[i]} - {names[j]}: "
                      f"{d:+.2f} +/- {se:.2f} over {len(diffs)} seeds  "
                      f"({wins}W/{len(diffs)-wins-ties}L/{ties}T)  -> {verdict}")

    if args.save:
        pathlib.Path(args.save).write_text(json.dumps(
            [{"config": r.config, "seed": r.seed, "score": r.score,
              "steps": r.steps, "calls": r.calls, "usd": r.usd,
              "ms": statistics.median(r.decide_ms) if r.decide_ms else 0,
              "greedy": r.greedy_hits / max(r.decisions, 1),
              "death": r.death} for r in results], indent=1))
        print(f"{DIM}per-seed results -> {args.save}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
