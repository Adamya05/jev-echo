"""Collect Jev's answers to the RAW BOARD, keeping the board itself.

The earlier dataset stored 28 precomputed features, which makes any student
trained on it incomparable to a model reading the board. This one stores the
board exactly as Jev sees it, so the student can consume the same input.

A strong local policy drives the game (free) so the states are worth
labelling; Jev labels every decision from the raw board.

    uv run python collect_raw.py --games 90
"""
from __future__ import annotations
import argparse, pathlib, random, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from sysone.snake import policy as P
from sysone.snake.game import Snake

MOVES = ("UP", "DOWN", "LEFT", "RIGHT")


def play(seed: int, max_steps: int) -> dict:
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from sysone.core import get_backend
    from sysone.snake.student import StudentBackend

    jev = get_backend("jev_or")
    driver = StudentBackend("results/student_prior.npz")
    g = Snake(seed=seed)
    boards, masks, targets = [], [], []
    usd = 0.0

    while g.alive and g.steps < max_steps:
        facts = g.options()
        if not facts:
            break
        if len(facts) == 1:
            g.step(next(iter(facts))); continue
        d = jev.decide(P.raw_state(g), {"move": P.raw_move_question(facts)})
        usd += d.usd
        p = {m: v for m, v in d["move"].probabilities.items() if m in facts}
        tot = sum(p.values())
        if tot > 0:
            boards.append(g.ascii_board().replace("\n", ""))
            masks.append([1.0 if m in facts else 0.0 for m in MOVES])
            targets.append([p.get(m, 0.0) / tot for m in MOVES])
        # The driver only plays safe moves; when none are left the game is over.
        # (With crashes allowed, options() is never empty, so it can't be the check.)
        safe = driver.predict(g)
        if not safe:
            break
        g.step(max(safe.items(), key=lambda kv: kv[1])[0])
    return {"boards": boards, "mask": masks, "target": targets,
            "score": g.score, "usd": usd}


def _job(a): return play(*a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=90)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=3000)
    ap.add_argument("--out", default="results/dataset_board.npz")
    a = ap.parse_args()

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        games = list(pool.map(_job, [(a.seed_offset + i, a.max_steps)
                                     for i in range(a.games)]))
    wall = time.perf_counter() - t0

    boards = [b for g in games for b in g["boards"]]
    data = {
        "boards": np.array(boards),
        "mask": np.array([m for g in games for m in g["mask"]], np.float32),
        "target": np.array([t for g in games for t in g["target"]], np.float32),
    }
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    usd = sum(g["usd"] for g in games)
    print(f"{len(boards):,} states -> {out}")
    print(f"  teacher score {np.mean([g['score'] for g in games]):.1f}")
    print(f"  wall {wall:.0f}s  ${usd:.3f}  {len(boards)/max(wall,1):.0f} states/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
