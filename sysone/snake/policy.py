"""State serialization and question design for Snake.

Two things are deliberate here:

1. The state is kept small. A 10x10 board plus a short JSON header lands
   around 400-500 input tokens, which is inside the 512-token context of
   the local laya-mlx English checkpoint. Build the demo at 20x20 and the
   hosted/local comparison stops being apples-to-apples.

2. Tiered goals (Goedecke). A strategy is re-chosen every STRATEGY_EVERY
   ticks and injected into the per-tick prompt. One fast model making a
   bounded choice every tick, under a slower-moving bounded choice, gets
   you planning behaviour without a planner.
"""

from __future__ import annotations

from typing import Any

from sysone.core import Choice, Noul
from sysone.snake.game import MoveFacts, Snake

STRATEGY_EVERY = 8

STRATEGIES = {
    "CHASE_FOOD": "Head straight for the food. Accept tight spaces to get there sooner.",
    "PLAY_SAFE": "Prefer open space and keeping a route to the tail. Reaching food can wait.",
    "FOLLOW_TAIL": "Stall deliberately: trail the tail in a loop until the board opens up.",
}


def build_state(game: Snake, strategy: str, compact: bool = False) -> dict[str, Any]:
    """Serialize the game for the model.

    `compact` drops the ASCII board. The per-move facts already encode
    everything the choice depends on, so the board is arguably ~300 tokens
    of redundancy -- but that is a claim worth measuring, not assuming.
    See bench_state.py.
    """
    state: dict[str, Any] = {
        "grid": f"{game.width}x{game.height}, walls on all four edges",
        "head": {"row": game.head[1], "col": game.head[0]},
        "food": {"row": game.food[1], "col": game.food[0]},
        "snake_length": len(game.body),
        "free_cells": game.width * game.height - len(game.body),
        "current_strategy": strategy,
    }
    if not compact:
        state["legend"] = "@ = head, o = body, * = food, . = empty"
        state["board"] = game.ascii_board()
    return state


def move_question(facts: dict[str, MoveFacts], strategy: str) -> Choice:
    return Choice(
        instructions=(
            "Choose the snake's next move. Every option listed is legal and "
            "survives this step -- the facts after each one are exact, already "
            f"computed, and can be trusted. Current strategy: {strategy}. "
            "Prefer moves that keep enough open space for the whole body and "
            "keep a route back to the tail; a shorter path to food is only "
            "worth it when the space is there."
        ),
        criteria={m: f.as_criterion() for m, f in facts.items()},
    )


def per_tick_questions(facts: dict[str, MoveFacts], strategy: str) -> dict[str, Any]:
    return {
        "move": move_question(facts, strategy),
        "trapped": Noul(
            instructions=(
                "Is the snake about to seal itself into a region of the board "
                "smaller than its own body, with no route back to its tail?"
            )
        ),
        "crowded": Noul(
            instructions=(
                "Has the board become crowded enough that survival now matters "
                "more than reaching the food?"
            )
        ),
    }


def strategy_question(current: str) -> Choice:
    return Choice(
        instructions=(
            f"The snake is currently playing {current}. Pick the strategy for "
            "the next several moves, given how much room is left on the board "
            "and how long the snake has grown."
        ),
        criteria=dict(STRATEGIES),
    )


# --------------------------------------------------------------------------
# Raw variant: no pre-chewed facts
# --------------------------------------------------------------------------

def raw_state(game: Snake) -> dict[str, Any]:
    """Just the board. No flood fill, no distances, nothing computed."""
    return {
        "legend": "@ = head, o = body, * = food, . = empty; walls on all edges",
        "board": game.ascii_board(),
        # Named explicitly. A bare [x, y] pair reads as (row, col) to a model,
        # which inverts every direction the criteria describe.
        "head": {"row": game.head[1], "col": game.head[0]},
        "food": {"row": game.food[1], "col": game.food[0]},
        "snake_length": len(game.body),
        "free_cells": game.width * game.height - len(game.body),
    }


DIRECTION_HINT = {
    "UP": "up (row - 1)", "DOWN": "down (row + 1)",
    "LEFT": "left (column - 1)", "RIGHT": "right (column + 1)",
}


def raw_move_question(facts: dict[str, MoveFacts]) -> Choice:
    """Bare directions. The model has to read the board itself.

    This is the control for "inherent intelligence": the chewed variant hands
    over flood-fill results and distances, so it measures judgement given
    facts. This one measures whether the model can derive the facts at all.
    """
    return Choice(
        instructions=(
            "Choose the snake's next move from the board. Avoid walls and the "
            "snake's own body, head towards the food, and do not move into a "
            "space too small to hold the snake."
        ),
        criteria={m: DIRECTION_HINT[m] for m in facts},
    )
