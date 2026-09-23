"""Collect distillation data from a teacher policy.

    uv run python -m sysone.snake.collect --games 150 --out results/dataset.npz

Logs both the raw Jev prior and the search-improved distribution at every
state, so one run trains both the prior-distilled and the search-distilled
student. Collection is the expensive half; training is cheap.

Forced moves are skipped -- there is no opinion to distil when only one
move survives, and including them would teach the student to be confident
about states it will never be asked to judge.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sysone.core import get_backend
from sysone.snake import features as F
from sysone.snake import search
from sysone.snake.game import Snake

BOLD, DIM, RESET = "\x1b[1m", "\x1b[2m", "\x1b[0m"


def play(seed: int, teacher: str, rollouts: int, max_steps: int) -> dict:
    pol = search.build(teacher, get_backend(), rollouts=rollouts)
    game = Snake(seed=seed)
    rng = random.Random(seed * 7919 + 13)

    feats, masks, priors, improved, lengths, steps = [], [], [], [], [], []
    usd = 0.0

    while game.alive and game.steps < max_steps:
        facts = game.survivable_moves()
        if not facts:
            break
        if len(facts) == 1:
            game.step(next(iter(facts)))
            continue

        st = pol.select(game, rng)
        usd += st.usd

        if st.prior and st.improved:
            feats.append(F.encode(game, facts))
            masks.append(F.mask(facts))
            priors.append(F.target(_norm(st.prior)))
            improved.append(F.target(_norm(st.improved)))
            lengths.append(len(game.body))
            steps.append(game.steps)

        game.step(st.move)

    return {
        "features": feats, "mask": masks, "prior": priors,
        "improved": improved, "length": lengths, "step": steps,
        "final_score": game.score, "usd": usd, "seed": seed,
    }


def _norm(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values())
    return {k: v / total for k, v in d.items()} if total > 0 else d


def _job(args: tuple) -> dict:
    return play(*args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--teacher", default="bestof", choices=["jev", "bestof"])
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=110)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=1000,
                    help="keep collection seeds disjoint from the 0-99 eval seeds")
    ap.add_argument("--out", default="results/dataset.npz")
    args = ap.parse_args()

    payload = [(args.seed_offset + i, args.teacher, args.rollouts, args.max_steps)
               for i in range(args.games)]
    print(f"\n{BOLD}collecting{RESET} {args.games} games with {args.teacher}"
          f"({args.rollouts}), seeds {args.seed_offset}-"
          f"{args.seed_offset + args.games - 1}\n")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        games = list(pool.map(_job, payload))
    wall = time.perf_counter() - t0

    data = {k: np.array([row for g in games for row in g[k]], dtype=np.float32)
            for k in ("features", "mask", "prior", "improved")}
    data["length"] = np.array([v for g in games for v in g["length"]], dtype=np.int32)
    data["step"] = np.array([v for g in games for v in g["step"]], dtype=np.int32)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    n = len(data["features"])
    scores = [g["final_score"] for g in games]
    usd = sum(g["usd"] for g in games)
    disagree = (data["prior"].argmax(1) != data["improved"].argmax(1)).mean()

    print(f"{BOLD}{n:,} states{RESET} from {args.games} games -> {out}")
    print(f"  teacher score   {np.mean(scores):.1f} +/- "
          f"{np.std(scores)/len(scores)**0.5:.2f}")
    print(f"  snake length    median {np.median(data['length']):.0f}, "
          f"p90 {np.percentile(data['length'], 90):.0f}, "
          f"max {data['length'].max()}")
    print(f"  search changed the argmax on {disagree:.1%} of states")
    print(f"  {DIM}wall {wall:.0f}s   ${usd:.3f}   {n/max(wall,1):.0f} states/s{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
