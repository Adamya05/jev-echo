"""Collect Laya's answers on a good state distribution.

The Jev dataset was gathered from strong play (best-of-N). To make the two
students comparable, Laya has to be asked about the *same kind of states* --
not the ones Laya reaches on its own, which are short-snake and degenerate
because Laya plays badly.

So: the J-student drives the game (free, plays like Jev), and Laya is asked
at every decision point. Same state distribution, different teacher.

    uv run python collect_laya.py --games 90
"""

from __future__ import annotations

import argparse
import pathlib
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sysone.snake import features as F
from sysone.snake import policy as jev_policy
from sysone.snake.game import Snake


def play(seed: int, max_steps: int) -> dict:
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from sysone.core import LayaBackend
    from sysone.snake.student import StudentBackend

    laya = LayaBackend()
    driver = StudentBackend("results/student_prior.npz")
    game = Snake(seed=seed)

    feats, masks, priors, lengths = [], [], [], []
    while game.alive and game.steps < max_steps:
        facts = game.survivable_moves()
        if not facts:
            break
        if len(facts) == 1:
            game.step(next(iter(facts)))
            continue

        d = laya.decide(
            jev_policy.build_state(game, "CHASE_FOOD", compact=True),
            {"move": jev_policy.move_question(facts, "CHASE_FOOD")})
        p = {m: v for m, v in d["move"].probabilities.items() if m in facts}
        total = sum(p.values())
        if total > 0:
            feats.append(F.encode(game, facts))
            masks.append(F.mask(facts))
            priors.append(F.target({m: v / total for m, v in p.items()}))
            lengths.append(len(game.body))

        game.step(max(driver.predict(game).items(), key=lambda kv: kv[1])[0])

    return {"features": feats, "mask": masks, "prior": priors,
            "length": lengths, "score": game.score}


def _job(a): return play(*a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=90)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/dataset_laya.npz")
    args = ap.parse_args()

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        games = list(pool.map(_job, [(2000 + i, args.max_steps)
                                     for i in range(args.games)]))
    wall = time.perf_counter() - t0

    data = {k: np.array([r for g in games for r in g[k]], dtype=np.float32)
            for k in ("features", "mask", "prior")}
    data["improved"] = data["prior"]          # no search teacher here
    data["length"] = np.array([v for g in games for v in g["length"]], dtype=np.int32)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    n = len(data["features"])
    print(f"{n:,} states -> {out}")
    print(f"  median snake length {np.median(data['length']):.0f}, "
          f"max {data['length'].max()}")
    print(f"  wall {wall:.0f}s, {n/max(wall,1):.1f} states/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
