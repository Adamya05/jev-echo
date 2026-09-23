"""One slow game against as many fast games as fit in the same wall clock.

Jev plays a single game to death. The local model plays game after game for
exactly as long as Jev took. The point is throughput, not latency: how many
games, and at what average score, fit inside one cloud game.

    uv run python marathon.py --probe
"""
from __future__ import annotations
import argparse, json, pathlib, random, time

from sysone.snake import search
from sysone.snake.game import Snake


def one_game(pol, seed: int, max_moves: int, keep_every: int = 1):
    """Time the decisions only.

    Building replay frames costs ~2 ms a move, which is ten times what Echo
    spends deciding. Charging that to the player would make the recording
    apparatus the thing being measured.
    """
    g = Snake(seed=seed); rng = random.Random(seed * 7919 + 13)
    frames = []
    spent = 0.0
    while g.alive and g.steps < max_moves:
        f = g.survivable_moves()
        if not f:
            break
        t = time.perf_counter()
        move = pol.select(g, rng).move
        spent += (time.perf_counter() - t) * 1000
        g.step(move)
        if g.steps % keep_every == 0:
            flat = g.ascii_board().replace("\n", "")
            frames.append({"t": round(spent, 2), "s": g.score, "l": len(g.body),
                           "m": g.steps, "h": flat.find("@"), "f": flat.find("*"),
                           "b": [i for i, c in enumerate(flat) if c == "o"]})
    return {"frames": frames, "score": g.score, "moves": g.steps,
            "ms": round(spent, 2), "alive": g.alive}


def marathon(cfg: str, budget_ms: float, seed0: int, keep_every: int,
             cap: int = 2000):
    """Play game after game until the wall-clock budget is spent."""
    pol = search.build(cfg, None)
    games, elapsed, n = [], 0.0, 0
    while elapsed < budget_ms and n < cap:
        r = one_game(pol, seed0 + n, 600, keep_every)
        r["t0"] = round(elapsed, 1)
        elapsed += r["ms"]; n += 1
        games.append(r)
    return games


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--max-moves", type=int, default=600)
    ap.add_argument("--out", default="results/marathon.json")
    a = ap.parse_args()
    if not a.probe:
        return run_full(a)

    jev = search.build("jev_raw", None)
    t0 = time.perf_counter()
    jg = one_game(jev, 0, a.max_moves)
    jev_ms = jg["ms"]
    print(f"jev: score {jg['score']}, {jg['moves']} moves, {jev_ms/1000:.1f}s")

    stu = search.build("board_student", None)
    n, total, elapsed = 0, 0, 0.0
    t = time.perf_counter()
    while elapsed < jev_ms:
        r = one_game(stu, 100 + n, a.max_moves)
        elapsed += r["ms"]; total += r["score"]; n += 1
        if a.probe and n >= 400:
            break
    print(f"student: {n} games in the same {elapsed/1000:.1f}s, "
          f"mean score {total/max(n,1):.1f}")
    print(f"  -> {n} games per jev game; "
          f"wall to generate: {(time.perf_counter()-t):.0f}s")
    return 0


def run_full(a) -> int:
    jev = one_game(search.build("jev_raw", None), 0, a.max_moves)
    jev["t0"] = 0.0
    budget = jev["ms"]
    print(f"jev        1 game   score {jev['score']:>3}  "
          f"{jev['moves']:>3} moves  {budget/1000:>6.1f}s")

    # The local panels run well past Jev's single game so they never freeze:
    # the whole point is that they keep going after the cloud model is done.
    out = {"budget_ms": budget, "jev": [jev]}
    for cfg, key, keep, mult in (("board_student", "student", 6, 6.0),
                                 ("board_budget_208", "search", 1, 6.0)):
        gs = marathon(cfg, budget * mult, 100, keep, cap=4000)
        tot = sum(g["score"] for g in gs)
        out[key] = gs
        print(f"{key:<10} {len(gs):>3} games  mean score "
              f"{tot/len(gs):>5.1f}  {sum(g['ms'] for g in gs)/1000:>6.1f}s  "
              f"{sum(len(g['frames']) for g in gs):>6,} frames")

    p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(",", ":")))
    print(f"\nwrote {p} ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
