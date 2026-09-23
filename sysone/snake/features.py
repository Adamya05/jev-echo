"""Feature encoding for the student (phase 2).

The ASCII board does not go in. `analyse()` already computes everything the
decision depends on -- flood fill, dead ends, distances -- so the student
gets those scalars directly. Handing an MLP a 100-character board and
asking it to rediscover flood fill would be teaching it to do badly, from
few samples, what a loop already does exactly.

Seven scalars per move, four moves in fixed order, 28 inputs.

Note what the fixed order costs. Jev takes the action space as an *input*:
a different option set with different natural-language facts every tick.
The student gets a fixed 4-way head and a mask. That is the specialization
trade -- it cannot generalize to a new action space, and in exchange it is
about five orders of magnitude faster.
"""

from __future__ import annotations

from sysone.snake.game import DIRECTIONS, MoveFacts, Snake

MOVES = ("UP", "DOWN", "LEFT", "RIGHT")   # fixed order; position carries meaning
PER_MOVE = 7
N_FEATURES = len(MOVES) * PER_MOVE

FEATURE_NAMES = [
    "available", "eats_food", "food_distance", "open_space",
    "room_for_body", "tail_reachable", "space_margin",
]


def encode(game: Snake, facts: dict[str, MoveFacts] | None = None) -> list[float]:
    """Game -> 28 floats, all roughly in [-1, 1]."""
    facts = facts if facts is not None else game.survivable_moves()
    cells = game.width * game.height
    span = game.width + game.height
    length = len(game.body)

    out: list[float] = []
    for m in MOVES:
        f = facts.get(m)
        if f is None:
            out.extend([0.0] * PER_MOVE)     # unavailable: masked at inference
            continue
        out.extend([
            1.0,
            1.0 if f.eats_food else 0.0,
            f.food_distance / span,
            f.open_space / cells,
            1.0 if f.room_for_body else 0.0,
            1.0 if f.keeps_tail_reachable else 0.0,
            (f.open_space - length) / cells,
        ])
    return out


def mask(facts: dict[str, MoveFacts]) -> list[float]:
    """1.0 for moves that exist this tick, 0.0 otherwise."""
    return [1.0 if m in facts else 0.0 for m in MOVES]


def target(distribution: dict[str, float]) -> list[float]:
    """A teacher's distribution over named moves -> a 4-vector in MOVES order.

    Soft targets, not argmax labels: the full distribution carries far more
    signal per sample, which is the entire reason distillation is cheaper
    than learning from scratch.
    """
    return [float(distribution.get(m, 0.0)) for m in MOVES]


def describe(vec: list[float]) -> str:
    rows = []
    for i, m in enumerate(MOVES):
        chunk = vec[i * PER_MOVE:(i + 1) * PER_MOVE]
        if chunk[0] == 0.0:
            rows.append(f"  {m:<6} unavailable")
            continue
        rows.append(f"  {m:<6} " + "  ".join(
            f"{n}={v:.2f}" for n, v in zip(FEATURE_NAMES[1:], chunk[1:])))
    return "\n".join(rows)
