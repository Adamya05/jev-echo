"""Replay recording for the crash branch: all three from seed 0, crashes allowed.

    SNAKE_ALLOW_CRASH=1 uv run python record_crash.py --budget 181
"""
import argparse, json, pathlib
from marathon import one_game, marathon
from sysone.snake import search
from sysone.snake.game import ALLOW_CRASH

assert ALLOW_CRASH, "run with SNAKE_ALLOW_CRASH=1"
ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=int, default=181)
a = ap.parse_args()

jev = one_game(search.build("jev_raw", None), 0, 600); jev["t0"] = 0.0
print(f"jev:  score {jev['score']}, {jev['moves']} moves, {jev['ms']/1000:.1f}s, alive={jev['alive']}")
sg = one_game(search.build(f"board_budget_{a.budget}", None), 0, 600); sg["t0"] = 0.0
print(f"search: score {sg['score']}, {sg['moves']} moves, {sg['ms']/1000:.1f}s, alive={sg['alive']}")
beat = max(jev["ms"], sg["ms"])
echo = marathon("board_student", beat * 1.35, 0, 6, cap=8000)
print(f"echo: {len(echo)} games, first-game score {echo[0]['score']}")
pathlib.Path("results/marathon_crash.json").write_text(json.dumps(
    {"budget_ms": jev["ms"], "jev": [jev], "search": [sg], "student": echo},
    separators=(",", ":")))
