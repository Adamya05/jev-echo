"""Find the position shown in "How they play" and record all three answers.

    SNAKE_ALLOW_CRASH=1 uv run python example_move.py      # one Jev call, ~$0.00002

Plays board 0 with the tree until it reaches a position where Echo is sure
of a move (P > 0.6) that the tree sees ends the game, then asks Jev about
that same board. Writes viz/example_move.json. The tree's depth depends on
timing, so a re-run can pick a slightly different position.
"""
import json, pathlib, random
from sysone.core import get_backend
from sysone.snake import policy as P, search
from sysone.snake.game import ALLOW_CRASH, Snake

assert ALLOW_CRASH, "run with SNAKE_ALLOW_CRASH=1: the page is the crashes-allowed game"

tree = search.build("board_tree_181", None)
g = Snake(seed=0)
while g.alive and g.steps < 600:
    facts = g.options()
    echo = tree.echo.predict(g)
    st = tree.select(g, random.Random(g.steps))
    pick = max(echo, key=echo.get)
    if len(facts) == 3 and g.score >= 5 and st.move != pick and echo[pick] > 0.6 \
            and st.visits["vals"][pick] < -1:
        break
    g.step(st.move)
else:
    raise SystemExit("no such position on this board")

d = get_backend("jev_or").decide(P.raw_state(g), {"move": P.raw_move_question(facts)})
state = P.raw_state(g)
ex = {"board": state["board"], "head": state["head"], "food": state["food"],
      "length": state["snake_length"],
      "jev": {m: round(v, 3) for m, v in d["move"].probabilities.items() if m in facts},
      "echo": {m: round(v, 3) for m, v in echo.items()},
      "search": {m: round(v, 3) for m, v in st.improved.items()},
      "search_vals": {m: round(v, 2) for m, v in st.visits["vals"].items()},
      "depth": st.visits["depth"], "boards": st.calls, "search_ms": round(st.ms),
      "note": f"Board 0, move {g.steps}, crashes allowed. Jev via OpenRouter; "
              "Echo and the tree locally. Made by example_move.py."}
pathlib.Path("viz/example_move.json").write_text(json.dumps(ex, indent=1))
print(f"move {g.steps}: jev {ex['jev']}  echo {ex['echo']}  tree {ex['search']}")
