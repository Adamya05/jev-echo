"""Record the replay shown on the page: all three players start from seed 0.

    uv run python record_replay.py                       # safety net on
    SNAKE_ALLOW_CRASH=1 uv run python record_replay.py   # safety net off

Jev and Echo + search play one game each. Echo plays game after game for a
little longer than the slower of the two, so it keeps going on the page.
Costs one Jev game (about $0.01).
"""
import argparse, json, pathlib
from marathon import marathon, one_game
from sysone.snake import search
from sysone.snake.game import ALLOW_CRASH

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=181 if ALLOW_CRASH else 197,
                help="Echo + search ms per move; match it to Jev's measured latency")
a = ap.parse_args()

jev = one_game(search.build("jev_raw", None), 0, 600)
srch = one_game(search.build(f"board_budget_{a.budget}", None), 0, 600)
jev["t0"] = srch["t0"] = 0.0
echo = marathon(search.build("board_student", None), max(jev["ms"], srch["ms"]) * 1.35, 0, 6)

out = pathlib.Path("results") / ("marathon_crash.json" if ALLOW_CRASH else "marathon.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"budget_ms": jev["ms"], "jev": [jev], "search": [srch],
                           "student": echo}, separators=(",", ":")))
print(f"jev {jev['score']}  echo+search {srch['score']}  echo {len(echo)} games -> {out}")
