"""Snake: pure game logic plus the analysis layer.

The split here is the entire lesson. Flood fill, dead-end detection and
distance are *arithmetic* -- they are cheap, exact, and a model that
guesses at them is strictly worse than a loop that computes them. So code
does all of that, and the model is handed a short list of legal moves with
their consequences already spelled out. Its only job is judgement:
given these four futures, which one.
"""

from __future__ import annotations

import copy
import os
import random
from collections import deque
from dataclasses import dataclass, field

DIRECTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

# When set, players are offered every legal move -- including ones that crash --
# instead of only the survivable ones. Off, behaviour is unchanged.
ALLOW_CRASH = os.environ.get("SNAKE_ALLOW_CRASH") == "1"


@dataclass
class MoveFacts:
    """Everything code can know about a candidate move, computed exactly."""
    move: str
    lands_on: tuple[int, int]
    fatal: bool
    eats_food: bool
    food_distance: int
    open_space: int          # free cells reachable after the move
    room_for_body: bool      # open_space >= snake length
    keeps_tail_reachable: bool

    def as_criterion(self) -> str:
        """Compact natural-language fact string handed to Jev."""
        bits = [f"moves to {self.lands_on}"]
        if self.eats_food:
            bits.append("EATS THE FOOD")
        bits.append(f"food then {self.food_distance} steps away")
        bits.append(f"{self.open_space} cells of open space")
        if not self.room_for_body:
            bits.append("NOT ENOUGH ROOM for the body - likely traps the snake")
        if not self.keeps_tail_reachable:
            bits.append("cannot reach own tail afterwards")
        return "; ".join(bits)


@dataclass
class Snake:
    width: int = 10
    height: int = 10
    body: deque[tuple[int, int]] = field(default_factory=deque)
    direction: str = "RIGHT"
    food: tuple[int, int] = (0, 0)
    score: int = 0
    steps: int = 0
    alive: bool = True
    seed: int = 7
    rng: random.Random = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random(self.seed)
        if not self.body:
            cx, cy = self.width // 2, self.height // 2
            self.body = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
            self.place_food()

    def clone(self) -> "Snake":
        """A full independent copy, including the RNG state.

        Carrying the RNG matters: a search that re-simulates food placement
        from a fresh generator is searching a different game than the one
        being played.
        """
        return copy.deepcopy(self)

    def state_key(self) -> tuple:
        """Identity of a position, for caching priors across a search."""
        return (tuple(self.body), self.food, self.direction)

    # -- geometry ----------------------------------------------------------

    @property
    def head(self) -> tuple[int, int]:
        return self.body[0]

    def in_bounds(self, c: tuple[int, int]) -> bool:
        return 0 <= c[0] < self.width and 0 <= c[1] < self.height

    def place_food(self) -> None:
        free = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.body
        ]
        self.food = self.rng.choice(free) if free else (-1, -1)

    def _flood(self, start: tuple[int, int], blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
        if not self.in_bounds(start) or start in blocked:
            return set()
        seen = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in DIRECTIONS.values():
                n = (x + dx, y + dy)
                if n not in seen and self.in_bounds(n) and n not in blocked:
                    seen.add(n)
                    stack.append(n)
        return seen

    # -- analysis ----------------------------------------------------------

    def legal_moves(self) -> list[str]:
        """Directions that are not an immediate reversal into our own neck."""
        return [m for m in DIRECTIONS if m != OPPOSITE[self.direction]]

    def analyse(self, move: str) -> MoveFacts:
        dx, dy = DIRECTIONS[move]
        hx, hy = self.head
        nxt = (hx + dx, hy + dy)

        eats = nxt == self.food
        body_after = deque(self.body)
        body_after.appendleft(nxt)
        if not eats:
            body_after.pop()  # tail vacates unless we grew

        fatal = not self.in_bounds(nxt) or nxt in list(self.body)[:-1]

        if fatal:
            return MoveFacts(move, nxt, True, eats, 999, 0, False, False)

        # The tail cell vacates on the next step, so it is not an obstacle
        # for the purpose of "can I still follow my own tail out of here".
        tail = body_after[-1]
        blocked = set(list(body_after)[1:-1])
        reachable = self._flood(nxt, blocked)

        return MoveFacts(
            move=move,
            lands_on=nxt,
            fatal=False,
            eats_food=eats,
            food_distance=abs(nxt[0] - self.food[0]) + abs(nxt[1] - self.food[1]),
            open_space=len(reachable),
            room_for_body=len(reachable) >= len(body_after),
            keeps_tail_reachable=tail in reachable or tail == nxt,
        )

    def options(self) -> dict[str, MoveFacts]:
        """The moves a player is offered: survivable ones, or every legal one."""
        if ALLOW_CRASH:
            return {m: self.analyse(m) for m in self.legal_moves()}
        return self.survivable_moves()

    def survivable_moves(self) -> dict[str, MoveFacts]:
        """Legal, non-fatal moves. The action space Jev actually chooses from."""
        facts = {m: self.analyse(m) for m in self.legal_moves()}
        return {m: f for m, f in facts.items() if not f.fatal}

    # -- stepping ----------------------------------------------------------

    def step(self, move: str) -> None:
        if move not in DIRECTIONS or move == OPPOSITE[self.direction]:
            move = self.direction
        facts = self.analyse(move)
        self.direction = move
        self.steps += 1

        if facts.fatal:
            self.alive = False
            return

        self.body.appendleft(facts.lands_on)
        if facts.eats_food:
            self.score += 1
            self.place_food()
        else:
            self.body.pop()

    # -- rendering ---------------------------------------------------------

    def ascii_board(self) -> str:
        cells = [["." for _ in range(self.width)] for _ in range(self.height)]
        for i, (x, y) in enumerate(self.body):
            cells[y][x] = "@" if i == 0 else "o"
        if self.in_bounds(self.food):
            fx, fy = self.food
            cells[fy][fx] = "*"
        return "\n".join("".join(row) for row in cells)
