"""Let Jev keep playing, like Echo does.

The replay held one Jev game, so Jev sat idle once it ended while Echo's
counter kept running. That flattered Echo by roughly 2x. Jev now restarts
too, and plays past the point where the demo pauses.
"""
import json, pathlib
from marathon import one_game
from sysone.snake import search

p = pathlib.Path("results/marathon.json")
d = json.loads(p.read_text())
beat = max(d["budget_ms"], d["search"][0]["t0"] + d["search"][0]["ms"])
target = beat * 1.35

jev = search.build("jev_raw", None)
games = d["jev"][:1]
t = games[0]["ms"]
seed = 1
while t < target:
    g = one_game(jev, seed, 600)
    g["t0"] = round(t, 2)
    t += g["ms"]
    games.append(g)
    print(f"jev game {len(games)}: seed {seed}, score {g['score']}, {g['ms']/1000:.1f}s")
    seed += 1
d["jev"] = games
p.write_text(json.dumps(d, separators=(",", ":")))
print(f"jev now covers {t/1000:.1f}s (beat {beat/1000:.1f}s)")
