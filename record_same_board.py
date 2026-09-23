"""Re-record the local players so every replay starts from Jev's board.

Jev's recorded game (seed 0) is kept as-is. Echo + search replays seed 0 with a
per-move budget matched to Jev's, and Echo's run of games begins at seed 0.
"""
import json, pathlib
from marathon import one_game, marathon
from sysone.snake import search

p = pathlib.Path("results/marathon.json")
d = json.loads(p.read_text())
jev = d["jev"][0]
print(f"jev: seed 0, score {jev['score']}, {jev['moves']} moves, {jev['ms']/1000:.1f}s")

sg = one_game(search.build("board_budget_197", None), 0, 600)
sg["t0"] = 0.0
print(f"echo+search: seed 0, score {sg['score']}, {sg['moves']} moves, {sg['ms']/1000:.1f}s")

beat = max(jev["ms"], sg["ms"])
echo = marathon("board_student", beat * 1.35, 0, 6, cap=5000)
print(f"echo: {len(echo)} games from seed 0, first game score {echo[0]['score']}")

d.update(jev=[jev], search=[sg], student=echo)
p.write_text(json.dumps(d, separators=(",", ":")))
rate = lambda gs: sum(g["moves"] for g in gs) / (sum(g["ms"] for g in gs) / 1000)
print(f"moves/s  jev {rate([jev]):.2f}  search {rate([sg]):.2f}  echo {rate(echo):,.0f}")
