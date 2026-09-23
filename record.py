"""Record games with real per-move timings, for side-by-side replay.

Each frame carries the board, the stats, and the wall-clock offset at which
it happened. The replay then advances each config on a shared clock, so the
speed difference is literal rather than described.

    uv run python record.py --config jev_self --seed 5 --moves 120
"""

from __future__ import annotations

import argparse, json, pathlib, random, time

from sysone.snake import search
from sysone.snake.game import Snake


def record(config: str, seed: int, max_moves: int) -> dict:
    pol = search.build(config, None)
    game = Snake(seed=seed)
    rng = random.Random(seed * 7919 + 13)

    frames = [{"t": 0.0, "board": game.ascii_board(), "len": len(game.body),
               "score": 0, "move": 0, "calls": 0}]
    t0 = time.perf_counter()
    calls = usd = 0

    while game.alive and game.steps < max_moves:
        facts = game.survivable_moves()
        if not facts:
            break
        st = pol.select(game, rng)
        calls += st.calls
        usd += st.usd
        game.step(st.move)
        frames.append({
            "t": round((time.perf_counter() - t0) * 1000, 1),
            "board": game.ascii_board(),
            "len": len(game.body), "score": game.score,
            "move": game.steps, "calls": calls,
        })

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "config": config, "name": pol.name, "seed": seed,
        "frames": frames, "score": game.score, "moves": game.steps,
        "alive": game.alive, "calls": calls, "usd": round(usd, 5),
        "total_ms": round(elapsed, 1),
        "ms_per_move": round(elapsed / max(game.steps, 1), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--moves", type=int, default=120)
    ap.add_argument("--out", default="results/recordings")
    ap.add_argument("--tag", action="store_true",
                    help="suffix the filename with the seed")
    a = ap.parse_args()

    r = record(a.config, a.seed, a.moves)
    d = pathlib.Path(a.out); d.mkdir(parents=True, exist_ok=True)
    name = f"{a.config}__s{a.seed}.json" if a.tag else f"{a.config}.json"
    (d / name).write_text(json.dumps(r))
    print(f"{r['name']:<20} score {r['score']:>3}  len {r['frames'][-1]['len']:>3}  "
          f"moves {r['moves']:>3}  calls {r['calls']:>5}  "
          f"{r['ms_per_move']:>8.1f} ms/move  total {r['total_ms']/1000:>6.1f}s  "
          f"${r['usd']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
