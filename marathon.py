"""Play Snake games with real per-move timings, for the replay on the page.

Only the time a player spends deciding is counted. Building the replay frames
costs ~2 ms a move, ten times what Echo spends deciding, so charging it to the
player would measure the recorder rather than the model.
"""
from __future__ import annotations

import random
import time

from sysone.snake.game import Snake


def one_game(pol, seed: int, max_moves: int, keep_every: int = 1) -> dict:
    """Play one game to the end. Keep a board snapshot every `keep_every` moves."""
    g = Snake(seed=seed)
    rng = random.Random(seed * 7919 + 13)
    frames, spent = [], 0.0
    while g.alive and g.steps < max_moves:
        if not g.options():
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


def marathon(pol, budget_ms: float, seed0: int, keep_every: int, cap: int = 8000) -> list[dict]:
    """Play game after game from seed0 upwards until budget_ms of play has passed."""
    games, elapsed = [], 0.0
    while elapsed < budget_ms and len(games) < cap:
        g = one_game(pol, seed0 + len(games), 600, keep_every)
        g["t0"] = round(elapsed, 1)
        elapsed += g["ms"]
        games.append(g)
    return games
