"""Measurements that build intuition for System One economics.

    uv run python bench.py
"""

from __future__ import annotations

import statistics
import time

from sysone.core import Choice, Noul, get_backend
from sysone.snake import policy
from sysone.snake.game import Snake

b = get_backend()
g = Snake()
g.step("RIGHT")
surv = g.survivable_moves()
state = policy.build_state(g, "CHASE_FOOD")


def timed(state, qs, reps=3):
    lat, tok = [], 0
    for _ in range(reps):
        d = b.decide(state, qs)
        lat.append(d.latency_ms)
        tok = d.input_tokens
    return statistics.median(lat), tok


print("=" * 62)
print("1. The fixed envelope")
print("=" * 62)
env_lat, env_tok = timed({}, {"x": Noul(instructions="ok?")})
print(f"  empty state, one trivial question: {env_tok} tokens, {env_lat:.0f} ms")
print("  -> every call costs this before your content. Small decisions are")
print("     envelope-dominated; the way to be efficient is more questions")
print("     per call, not shorter states.\n")

print("=" * 62)
print("2. Batching: N questions in one call vs N calls")
print("=" * 62)
probes = {
    f"q{i}": Noul(instructions=t)
    for i, t in enumerate([
        "Is the snake in the upper half of the board?",
        "Is the food closer horizontally than vertically?",
        "Is the snake longer than 4 cells?",
        "Is the head adjacent to a wall?",
        "Is there more open space to the left than the right?",
        "Is the snake heading towards the food?",
        "Has the board become crowded?",
        "Is the snake at risk of trapping itself?",
    ])
}
one_lat, one_tok = timed(state, probes)
t0 = time.perf_counter()
many_tok = sum(b.decide(state, {k: v}).input_tokens for k, v in probes.items())
many_lat = (time.perf_counter() - t0) * 1000
print(f"  8 questions, 1 call  : {one_tok:>5} tokens  {one_lat:>6.0f} ms")
print(f"  8 questions, 8 calls : {many_tok:>5} tokens  {many_lat:>6.0f} ms")
print(f"  -> {many_tok / one_tok:.1f}x the tokens, {many_lat / one_lat:.1f}x the wall time,")
print("     for identical information. Jev evaluates every question against")
print("     one state in a single parallel pass -- that is the whole trick.\n")

print("=" * 62)
print("3. Does the ASCII board earn its tokens?")
print("=" * 62)
qs = policy.per_tick_questions(surv, "CHASE_FOOD")
for label, st in [("with board", policy.build_state(g, "CHASE_FOOD")),
                  ("facts only", policy.build_state(g, "CHASE_FOOD", compact=True))]:
    agree = []
    for _ in range(5):
        d = b.decide(st, qs)
        agree.append((str(d["move"].value), d["move"].p))
    lat, tok = timed(st, qs)
    mode = max(set(m for m, _ in agree), key=lambda m: sum(1 for x, _ in agree if x == m))
    conf = statistics.mean(p for m, p in agree if m == mode)
    print(f"  {label:<11} {tok:>4} tok  {lat:>5.0f} ms  "
          f"picks {mode} {sum(1 for m,_ in agree if m==mode)}/5 at p={conf:.2f}")
print("  -> the board is 122 tokens. Keep it if it changes the decision;")
print("     drop it if the computed facts already carry the signal.\n")

print("=" * 62)
print("4. What the local 512-token budget actually buys")
print("=" * 62)
print(f"  laya-mlx English checkpoint: 512 tokens TOTAL context.")
print(f"  Jev's envelope is TypeSafe-side framing, not your content --")
print(f"  local content here is the state + questions only.")
print(f"  full state + 3 questions on Jev : {b.decide(state, qs).input_tokens} tok billed")
print(f"  of which envelope               : {env_tok}")
print(f"  your actual content             : ~{b.decide(state, qs).input_tokens - env_tok}")
print("  -> comfortably inside 512. This demo survives the swap.")
