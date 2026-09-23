"""Search on top of a System One prior.

Jev's `Choice` distribution is a categorical policy head, pi(a|s). It is a
usable *prior* and nothing more: RLCD trains for calibration, not for
return maximization, so `confidence` is not a value estimate and must
never be read as one. There is also no credit assignment anywhere in the
API -- every call is a stateless one-step judgment -- so there is no
learning loop to inherit.

Which leaves two ways to do better than the black box: improve a copy of
it (phase 2), or improve its output with search (here).

The binding constraint is money and milliseconds. A prior at every node
is unaffordable at ~160 ms per call, so:

  * exactly one Jev call per move, at the root only
  * deeper nodes get a cheap heuristic prior read off MoveFacts
  * value comes from simulator rollouts, not from the model

UNIFORM_MCTS is the control. Without it, any gain from JEV_MCTS could be
the search rather than the prior, and the experiment says nothing.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from sysone.core import Backend, Choice
from sysone.snake import policy as jev_policy
from sysone.snake.game import ALLOW_CRASH, MoveFacts, Snake

ECHO = os.environ.get("SNAKE_ECHO") or (
    "results/student_board_crash.npz" if ALLOW_CRASH else "results/student_board.npz")

C_PUCT = 1.4
ROLLOUT_DEPTH = 40
ROLLOUT_TEMP = 0.7
TARGET_TEMP = 0.1


# --------------------------------------------------------------------------
# Cheap prior: everything MoveFacts already knows, softmaxed
# --------------------------------------------------------------------------

def heuristic_scores(game: Snake, facts: dict[str, MoveFacts]) -> dict[str, float]:
    cells = game.width * game.height
    span = game.width + game.height
    out = {}
    for m, f in facts.items():
        s = 1.5 if f.eats_food else 0.0
        s += 1.0 * (f.open_space / cells)
        s += 0.8 if f.room_for_body else -1.5
        s += 0.5 if f.keeps_tail_reachable else -0.8
        s -= 2.2 * (f.food_distance / span)
        out[m] = s
    return out


def softmax(scores: dict[str, float], temp: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    hi = max(scores.values())
    exp = {k: math.exp((v - hi) / max(temp, 1e-6)) for k, v in scores.items()}
    total = sum(exp.values()) or 1.0
    return {k: v / total for k, v in exp.items()}


def heuristic_prior(game: Snake, facts: dict[str, MoveFacts] | None = None
                    ) -> dict[str, float]:
    facts = facts if facts is not None else game.options()
    return softmax(heuristic_scores(game, facts))


def uniform_prior(game: Snake, facts: dict[str, MoveFacts] | None = None
                  ) -> dict[str, float]:
    facts = facts if facts is not None else game.options()
    n = len(facts) or 1
    return {m: 1.0 / n for m in facts}


# --------------------------------------------------------------------------
# Rollouts supply the value signal
# --------------------------------------------------------------------------

def rollout(game: Snake, rng: random.Random, depth: int = ROLLOUT_DEPTH,
            policy=None) -> float:
    """Play out with the heuristic policy; return a value in [0, 1].

    The distance term is not decoration. Roughly half of all rollouts eat
    nothing, so a value built only on food and survival sits pinned at its
    floor and the between-move gap drops below the rollout noise -- which
    is enough to make best-of-N actively worse than the prior it is
    re-ranking. Ending closer to the food is the dense signal that keeps
    every rollout informative.
    """
    g = game.clone()
    start = g.score
    steps = 0
    while g.alive and steps < depth:
        facts = g.options()
        if not facts:
            break
        probs = (policy(g, facts) if policy is not None
                 else softmax(heuristic_scores(g, facts), ROLLOUT_TEMP))
        moves, weights = list(probs), list(probs.values())
        if not moves:
            break
        g.step(rng.choices(moves, weights=weights, k=1)[0])
        steps += 1

    food = g.score - start
    span = g.width + g.height
    end_dist = abs(g.head[0] - g.food[0]) + abs(g.head[1] - g.food[1]) if g.alive else span
    return (0.55 * min(food, 3) / 3
            + 0.20 * (steps / depth)
            + 0.25 * (1.0 - min(end_dist, span) / span))


# --------------------------------------------------------------------------
# PUCT
# --------------------------------------------------------------------------

@dataclass
class Node:
    game: Snake
    prior: dict[str, float]
    N: dict[str, int] = field(default_factory=dict)
    W: dict[str, float] = field(default_factory=dict)
    children: dict[str, "Node | None"] = field(default_factory=dict)
    visits: int = 0

    def __post_init__(self) -> None:
        for m in self.prior:
            self.N.setdefault(m, 0)
            self.W.setdefault(m, 0.0)
            self.children.setdefault(m, None)

    def q(self, m: str) -> float:
        return self.W[m] / self.N[m] if self.N[m] else 0.0

    def select(self, c_puct: float) -> str:
        root_n = math.sqrt(max(self.visits, 1))
        best, best_u = None, -1e18
        for m, p in self.prior.items():
            u = self.q(m) + c_puct * p * root_n / (1 + self.N[m])
            if u > best_u:
                best, best_u = m, u
        return best


def mcts(root_game: Snake, root_prior: dict[str, float], iterations: int,
         rng: random.Random, c_puct: float = C_PUCT,
         depth: int = ROLLOUT_DEPTH,
         prior_cache: dict[tuple, dict[str, float]] | None = None
         ) -> tuple[str | None, dict[str, int]]:
    """Returns the most-visited root move and the root visit counts.

    Visit counts, not Q. Visits are the robust MCTS output; a child with
    one lucky rollout can hold the best Q all search.
    """
    if not root_prior:
        return None, {}
    cache = prior_cache if prior_cache is not None else {}
    root = Node(root_game, root_prior)

    for _ in range(iterations):
        node, path = root, []

        while True:
            move = node.select(c_puct)
            if move is None:
                value = 0.0
                break
            path.append((node, move))

            child = node.children[move]
            if child is None:
                nxt = node.game.clone()
                nxt.step(move)
                if not nxt.alive:
                    value = 0.0
                    break
                key = nxt.state_key()
                if key not in cache:
                    cache[key] = heuristic_prior(nxt)
                node.children[move] = Node(nxt, cache[key])
                value = rollout(nxt, rng, depth)
                break

            if not child.prior:          # dead end: boxed in
                value = 0.0
                break
            node = child

        for n, m in path:
            n.visits += 1
            n.N[m] += 1
            n.W[m] += value

    counts = dict(root.N)
    best = max(counts, key=lambda m: (counts[m], root.q(m)))
    return best, counts


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

@dataclass
class Step:
    move: str
    calls: int = 0
    tokens: int = 0
    usd: float = 0.0
    ms: float = 0.0
    prior: dict[str, float] = field(default_factory=dict)
    visits: dict[str, int] = field(default_factory=dict)
    improved: dict[str, float] = field(default_factory=dict)
    """The policy's own distribution after search -- the distillation target.

    For JevOnly this is just the prior. For best-of-N it is a softmax over
    the blended candidate scores; for MCTS it would be normalized visit
    counts. Logging both this and `prior` means one collection run yields
    training data for a prior-distilled student and a search-distilled one.
    """


def jev_prior(backend: Backend, game: Snake, facts: dict[str, MoveFacts],
              cache: dict[tuple, Any], compact: bool = False
              ) -> tuple[dict[str, float], Step]:
    """One call. Root only. Cached by position.

    `compact` drops the ASCII board. Required for laya, whose English
    checkpoint caps at 512 tokens -- the full state tokenizes to 573 and
    would be silently truncated.
    """
    key = game.state_key()
    if key in cache:
        prior, st = cache[key]
        return prior, Step(move="", calls=0, prior=prior)

    state = jev_policy.build_state(game, "CHASE_FOOD", compact=compact)
    d = backend.decide(state, {"move": jev_policy.move_question(facts, "CHASE_FOOD")})
    prior = {m: p for m, p in d["move"].probabilities.items() if m in facts}
    total = sum(prior.values())
    prior = {m: p / total for m, p in prior.items()} if total > 0 else uniform_prior(game, facts)
    st = Step(move=str(d["move"].value), calls=1, tokens=d.input_tokens,
              usd=d.usd, ms=d.latency_ms, prior=prior)
    cache[key] = (prior, st)
    return prior, st


class JevOnly:
    """Take the prior's argmax and play it. Works for any hosted backend."""

    def __init__(self, backend: Backend, compact: bool = False,
                 label: str = "JEV") -> None:
        self.backend, self.cache = backend, {}
        self.compact = compact
        self.name = label

    def select(self, game: Snake, rng: random.Random) -> Step:
        facts = game.options()
        if not facts:
            return Step(move=game.direction)
        if len(facts) == 1:
            return Step(move=next(iter(facts)))
        prior, st = jev_prior(self.backend, game, facts, self.cache, self.compact)
        st.move = max(prior, key=prior.get)
        st.improved = dict(prior)
        return st


class JevBestOfN:
    """Jev proposes, one ply of simulation re-ranks. One call, no tree.

    A trap worth recording. The obvious version -- take the prior's top N,
    simulate each, play the best -- is degenerate here: the survivable
    action space is 3 moves or fewer in 100% of positions, so N=3 selects
    *every* move and the prior is discarded entirely. That config is not
    "prior plus search", it is a pure rollout policy, and it scores like
    one (2.5 against 12.2 for the prior alone).

    So the prior has to stay in the ranking, which is what PUCT does at
    every visit and what a one-shot argmax over rollout values does not.
    `prior_weight` is tuned against the observed rollout gap of ~0.1:
    large enough that the prior biases the ranking, small enough that
    strong simulation evidence can still overrule it.
    """
    def __init__(self, backend: Backend, n: int = 3, rollouts: int = 16,
                 prior_weight: float = 0.25) -> None:
        self.backend, self.n, self.rollouts = backend, n, rollouts
        self.prior_weight = prior_weight
        self.cache: dict[tuple, Any] = {}
        self.name = f"JEV+BESTOF({rollouts})"

    def select(self, game: Snake, rng: random.Random) -> Step:
        facts = game.options()
        if not facts:
            return Step(move=game.direction)
        if len(facts) == 1:
            return Step(move=next(iter(facts)))

        prior, st = jev_prior(self.backend, game, facts, self.cache)
        candidates = sorted(prior, key=prior.get, reverse=True)[: self.n]

        t0 = time.perf_counter()
        blended: dict[str, float] = {}
        best, best_s = None, -1e18
        for m in candidates:
            nxt = game.clone()
            nxt.step(m)
            v = 0.0 if not nxt.alive else (
                sum(rollout(nxt, rng) for _ in range(self.rollouts)) / self.rollouts
            )
            score = v + self.prior_weight * prior.get(m, 0.0)
            blended[m] = score
            if score > best_s:
                best, best_s = m, score
        st.ms += (time.perf_counter() - t0) * 1000   # rollouts are not free
        st.move = best or candidates[0]
        st.visits = {m: self.rollouts for m in candidates}
        # Blended scores span ~0.1; TARGET_TEMP turns that into a target with
        # roughly the sharpness of the prior rather than a one-hot.
        st.improved = softmax(blended, TARGET_TEMP)
        return st


class MCTSPolicy:
    """PUCT with a pluggable root prior. Jev or uniform -- that is the A/B."""

    def __init__(self, backend: Backend | None, iterations: int = 160,
                 use_jev: bool = True) -> None:
        self.backend, self.iterations, self.use_jev = backend, iterations, use_jev
        self.cache: dict[tuple, Any] = {}
        self.prior_cache: dict[tuple, dict[str, float]] = {}
        self.name = f"JEV+MCTS({iterations})" if use_jev else f"UNIFORM+MCTS({iterations})"

    def select(self, game: Snake, rng: random.Random) -> Step:
        facts = game.options()
        if not facts:
            return Step(move=game.direction)
        if len(facts) == 1:
            return Step(move=next(iter(facts)))

        if self.use_jev:
            prior, st = jev_prior(self.backend, game, facts, self.cache)
        else:
            prior, st = uniform_prior(game, facts), Step(move="")

        t0 = time.perf_counter()
        move, counts = mcts(game, prior, self.iterations, rng,
                            prior_cache=self.prior_cache)
        st.ms += (time.perf_counter() - t0) * 1000
        st.move = move or max(prior, key=prior.get)
        st.prior, st.visits = prior, counts
        total = sum(counts.values()) or 1
        st.improved = {m: c / total for m, c in counts.items()}
        return st


class LayaPolicy:
    """Laya with a prompt shaped for laya, not for Jev.

    Measured on 40 real positions: our Jev-format prompt (199 tok) picks the
    food-nearest move 52% of the time; a bare facts prompt (100 tok) picks it
    70% of the time. Benchmarking a model on another model's prompt format
    measures the prompt as much as the model, so this gives laya its own.
    """

    name = "LAYA(tuned)"

    def __init__(self) -> None:
        from sysone.core import LayaBackend
        self.backend = LayaBackend()

    def select(self, game: Snake, rng: random.Random) -> Step:
        facts = game.options()
        if not facts:
            return Step(move=game.direction)
        if len(facts) == 1:
            return Step(move=next(iter(facts)))

        state = (f"Snake length {len(game.body)}. Head at {game.head}. "
                 f"Food at {game.food}.")
        q = Choice(
            instructions="Pick the move that reaches the food soonest while "
                         "keeping space to move.",
            criteria={m: f.as_criterion() for m, f in facts.items()})
        d = self.backend.decide(state, {"move": q})
        a = d["move"]
        prior = {m: p for m, p in a.probabilities.items() if m in facts}
        move = str(a.value) if str(a.value) in facts else max(
            prior, key=prior.get) if prior else game.direction
        return Step(move=move, ms=d.latency_ms, prior=prior, improved=prior)


class StudentSearch:
    """Search where BOTH the prior and the rollouts come from the student.

    This is the affordable version of "use Jev for the rollouts too".
    Doing it with Jev directly is out of reach: 3 candidates x 16 rollouts
    x 40 steps is ~1,920 calls for a single move, five minutes at 165 ms.
    The student is a distilled Jev at 0.43 ms, so the same shape of search
    costs milliseconds instead.

    Worth doing only now. Under the old 110-step cap the snake never grew
    long enough to reach a state where judgement mattered, so better
    rollouts had nothing to be better about. Uncapped, 14-38% of decisions
    are tight.
    """

    def __init__(self, path: str, label: str, n: int = 3, rollouts: int = 8,
                 prior_weight: float = 0.25, depth: int = 30) -> None:
        import mlx.core as mx
        from sysone.snake.student import StudentBackend
        mx.set_default_device(mx.cpu)
        self.backend = StudentBackend(path)
        self.n, self.rollouts, self.depth = n, rollouts, depth
        self.prior_weight = prior_weight
        self.name = f"STUDENT+SEARCH({label},{rollouts})"

    def _policy(self, game: Snake, facts) -> dict[str, float]:
        return self.backend.predict(game)

    def select(self, game: Snake, rng: random.Random) -> Step:
        facts = game.options()
        if not facts:
            return Step(move=game.direction)
        if len(facts) == 1:
            return Step(move=next(iter(facts)))

        t0 = time.perf_counter()
        prior = self.backend.predict(game)
        candidates = sorted(prior, key=prior.get, reverse=True)[: self.n]

        best, best_s, blended = None, -1e18, {}
        for m in candidates:
            nxt = game.clone()
            nxt.step(m)
            v = 0.0 if not nxt.alive else (
                sum(rollout(nxt, rng, self.depth, self._policy)
                    for _ in range(self.rollouts)) / self.rollouts)
            score = v + self.prior_weight * prior.get(m, 0.0)
            blended[m] = score
            if score > best_s:
                best, best_s = m, score

        return Step(move=best or candidates[0],
                    ms=(time.perf_counter() - t0) * 1000,
                    prior=prior, improved=softmax(blended, TARGET_TEMP))


class BoardStudent:
    """The 87 KB conv student, reading the same raw board Jev reads."""

    name = "STUDENT(board)"

    def __init__(self, path: str = ECHO) -> None:
        from sysone.snake.boardnet import BoardBackend
        self.backend = BoardBackend(path)

    def select(self, game: Snake, rng: random.Random) -> Step:
        facts = game.options()
        if not facts:
            return Step(move=game.direction)
        if len(facts) == 1:
            return Step(move=next(iter(facts)))
        t0 = time.perf_counter()
        p = self.backend.predict(game)
        ms = (time.perf_counter() - t0) * 1000
        move = max(p, key=p.get) if p else game.direction
        return Step(move=move, calls=1, ms=ms, prior=p, improved=p)


class BudgetSearch:
    """Same wall-clock per move as one Jev call -- spent on search instead.

    Jev's 208 ms buys exactly one decision. The same 208 ms buys the board
    student roughly 900 forward passes, so it can simulate instead of just
    reacting. Identical input, identical time budget; the only difference is
    what each can afford to do inside it.

    The budget is enforced against the clock, not a rollout count, so this
    stays honest if the hardware changes.
    """

    def __init__(self, budget_ms: float = 208.0, n: int = 3, depth: int = 8,
                 prior_weight: float = 0.25,
                 path: str = ECHO) -> None:
        from sysone.snake.boardnet import BoardBackend
        self.backend = BoardBackend(path)
        self.budget, self.n, self.depth = budget_ms, n, depth
        self.prior_weight = prior_weight
        self.name = f"STUDENT(board)+search@{int(budget_ms)}ms"

    def _policy(self, game: Snake, facts) -> dict[str, float]:
        return self.backend.predict(game)

    def select(self, game: Snake, rng: random.Random) -> Step:
        facts = game.options()
        if not facts:
            return Step(move=game.direction)
        if len(facts) == 1:
            return Step(move=next(iter(facts)))

        t0 = time.perf_counter()
        prior = self.backend.predict(game)
        cands = sorted(prior, key=prior.get, reverse=True)[: self.n]
        nxt = {}
        for m in cands:
            g2 = game.clone(); g2.step(m)
            nxt[m] = g2 if g2.alive else None

        tot = {m: 0.0 for m in cands}
        cnt = {m: 0 for m in cands}
        calls = 1
        while (time.perf_counter() - t0) * 1000 < self.budget:
            for m in cands:
                if (time.perf_counter() - t0) * 1000 >= self.budget:
                    break
                if nxt[m] is None:
                    cnt[m] = max(cnt[m], 1); continue
                tot[m] += rollout(nxt[m], rng, self.depth, self._policy)
                cnt[m] += 1
                calls += self.depth

        best, best_s, blended = None, -1e18, {}
        for m in cands:
            v = tot[m] / cnt[m] if cnt[m] else 0.0
            sc = v + self.prior_weight * prior.get(m, 0.0)
            blended[m] = sc
            if sc > best_s:
                best, best_s = m, sc
        return Step(move=best or cands[0], calls=calls,
                    ms=(time.perf_counter() - t0) * 1000,
                    prior=prior, improved=softmax(blended, TARGET_TEMP),
                    visits=dict(cnt))


class SelfSearch:
    """One model does everything: the prior AND the rollouts.

    This is the honest "pure Jev vs pure local" comparison -- no distilled
    student anywhere in the loop. It is only affordable with a shallow,
    narrow search, because every simulated step is a real model call:
    1 prior + n_candidates x rollouts x depth calls per move.
    """

    def __init__(self, which: str, label: str, n: int = 2,
                 rollouts: int = 2, depth: int = 5,
                 prior_weight: float = 0.25) -> None:
        if which == "student":
            import mlx.core as mx
            from sysone.snake.student import StudentBackend
            mx.set_default_device(mx.cpu)
            self.backend = StudentBackend("results/student_prior.npz")
        else:
            from sysone.core import get_backend
            self.backend = get_backend(which)
        self.n, self.rollouts, self.depth = n, rollouts, depth
        self.prior_weight = prior_weight
        self.name = label

    def _ask(self, game: Snake, facts) -> tuple[dict[str, float], float, int, float]:
        state = game if hasattr(self.backend, "predict") else jev_policy.raw_state(game)
        d = self.backend.decide(state,
                                {"move": jev_policy.raw_move_question(facts)})
        p = {m: v for m, v in d["move"].probabilities.items() if m in facts}
        t = sum(p.values())
        p = {m: v / t for m, v in p.items()} if t > 0 else uniform_prior(game, facts)
        return p, d.latency_ms, d.input_tokens, d.usd

    def _policy(self, game: Snake, facts) -> dict[str, float]:
        return self._ask(game, facts)[0]

    def select(self, game: Snake, rng: random.Random) -> Step:
        facts = game.options()
        if not facts:
            return Step(move=game.direction)
        if len(facts) == 1:
            return Step(move=next(iter(facts)))

        t0 = time.perf_counter()
        prior, _, tok, usd = self._ask(game, facts)
        calls = 1
        candidates = sorted(prior, key=prior.get, reverse=True)[: self.n]

        best, best_s, blended = None, -1e18, {}
        for m in candidates:
            nxt = game.clone()
            nxt.step(m)
            if not nxt.alive:
                blended[m] = 0.0
                continue
            vals = []
            for _ in range(self.rollouts):
                vals.append(rollout(nxt, rng, self.depth, self._policy))
                calls += self.depth
            v = sum(vals) / len(vals)
            score = v + self.prior_weight * prior.get(m, 0.0)
            blended[m] = score
            if score > best_s:
                best, best_s = m, score

        return Step(move=best or candidates[0], calls=calls,
                    tokens=tok * calls, usd=usd * calls,
                    ms=(time.perf_counter() - t0) * 1000,
                    prior=prior, improved=softmax(blended, TARGET_TEMP))


class RawPolicy:
    """One call per move, board only, no computed facts."""

    def __init__(self, which: str) -> None:
        from sysone.core import get_backend
        self.backend = get_backend(which)
        self.name = {"jev_or": "JEV(raw)", "laya_fast": "LAYA(raw)"}.get(which, which)

    def select(self, game: Snake, rng: random.Random) -> Step:
        facts = game.options()
        if not facts:
            return Step(move=game.direction)
        if len(facts) == 1:
            return Step(move=next(iter(facts)))
        d = self.backend.decide(jev_policy.raw_state(game),
                                {"move": jev_policy.raw_move_question(facts)})
        a = d["move"]
        prior = {m: p for m, p in a.probabilities.items() if m in facts}
        move = str(a.value) if str(a.value) in facts else (
            max(prior, key=prior.get) if prior else game.direction)
        return Step(move=move, calls=1, tokens=d.input_tokens, usd=d.usd,
                    ms=d.latency_ms, prior=prior, improved=prior)


class HybridSearch:
    """Jev picks the prior; the student runs the rollouts.

    Rollouts driven by Jev itself are unaffordable -- 3 candidates x 8
    rollouts x 30 steps is ~720 calls for one move. But the rollout policy
    does not have to be the same model as the prior. The student is a
    distilled Jev at 0.14 ms, so this costs exactly one Jev call per move,
    the same as playing Jev straight, and buys the search that was worth
    +60% to the student.

    It also isolates the variable: against STUDENT+SEARCH, the only
    difference is where the prior comes from.
    """

    def __init__(self, path: str = "results/student_prior.npz",
                 n: int = 3, rollouts: int = 8, prior_weight: float = 0.25,
                 depth: int = 30, prior_model: str = "jev",
                 rollout_label: str = "J") -> None:
        import mlx.core as mx
        from sysone.core import JevBackend, LayaBackend
        from sysone.snake.student import StudentBackend
        mx.set_default_device(mx.cpu)
        self.compact = prior_model == "laya"      # laya caps at 512 tokens
        self.jev = (LayaBackend() if prior_model == "laya"
                    else JevBackend(via="openrouter"))
        self.student = StudentBackend(path)
        self.n, self.rollouts, self.depth = n, rollouts, depth
        self.prior_weight = prior_weight
        self.cache: dict[tuple, Any] = {}
        self.name = f"{prior_model.upper()}+roll:{rollout_label}"

    def _policy(self, game: Snake, facts) -> dict[str, float]:
        return self.student.predict(game)

    def select(self, game: Snake, rng: random.Random) -> Step:
        facts = game.options()
        if not facts:
            return Step(move=game.direction)
        if len(facts) == 1:
            return Step(move=next(iter(facts)))

        prior, st = jev_prior(self.jev, game, facts, self.cache, self.compact)
        candidates = sorted(prior, key=prior.get, reverse=True)[: self.n]

        t0 = time.perf_counter()
        best, best_s, blended = None, -1e18, {}
        for m in candidates:
            nxt = game.clone()
            nxt.step(m)
            v = 0.0 if not nxt.alive else (
                sum(rollout(nxt, rng, self.depth, self._policy)
                    for _ in range(self.rollouts)) / self.rollouts)
            score = v + self.prior_weight * prior.get(m, 0.0)
            blended[m] = score
            if score > best_s:
                best, best_s = m, score
        st.ms += (time.perf_counter() - t0) * 1000
        st.move = best or candidates[0]
        st.improved = softmax(blended, TARGET_TEMP)
        return st


class StudentPolicy:
    """The distilled specialist, playing on its own. No calls, no rollouts."""

    def __init__(self, path: str, label: str) -> None:
        import mlx.core as mx
        from sysone.snake.student import StudentBackend
        mx.set_default_device(mx.cpu)      # measured 10x faster at this size
        self.backend = StudentBackend(path)
        self.name = f"STUDENT({label})"

    def select(self, game: Snake, rng: random.Random) -> Step:
        facts = game.options()
        if not facts:
            return Step(move=game.direction)
        if len(facts) == 1:
            return Step(move=next(iter(facts)))
        t0 = time.perf_counter()
        probs = self.backend.predict(game)
        ms = (time.perf_counter() - t0) * 1000
        move = max(probs, key=probs.get) if probs else game.direction
        return Step(move=move, calls=1, ms=ms, prior=probs, improved=probs)


def build(name: str, backend: Backend | None, iterations: int = 160,
          rollouts: int = 16):
    if name.startswith("studentsearch_"):
        label = name.removeprefix("studentsearch_")
        return StudentSearch(f"results/student_{label}.npz", label, rollouts=rollouts)
    if name.startswith("student_"):
        label = name.removeprefix("student_")
        return StudentPolicy(f"results/student_{label}.npz", label)
    if name == "jev":
        return JevOnly(backend)
    if name.startswith("self_"):
        _, who, r = name.split("_")          # self_<model>_<rollouts>
        spec = {"jev": ("jev_or", "JEV + JEV sims"),
                "laya": ("laya_fast", "LAYA + LAYA sims"),
                "student": ("student", "STUDENT + sims")}[who]
        return SelfSearch(spec[0], f"{spec[1]} x{r}", rollouts=int(r))
    if name == "student_solo":
        return StudentPolicy("results/student_prior.npz", "prior")
    if name == "board_student":
        return BoardStudent()
    if name.startswith("board_budget"):
        ms = name.split("_")[-1]
        return BudgetSearch(budget_ms=float(ms) if ms.isdigit() else 208.0)
    if name == "jev_self":
        return SelfSearch("jev_or", "JEV + JEV sims")
    if name == "laya_self":
        return SelfSearch("laya_fast", "LAYA + LAYA sims")
    if name == "jev_raw":
        return RawPolicy("jev_or")
    if name == "laya_raw":
        return RawPolicy("laya_fast")
    if name == "laya_fast":
        from sysone.core import get_backend
        return JevOnly(get_backend("laya_fast"), compact=True, label="LAYA(chewed)")
    if name.startswith("search_"):
        _, prior_model, roll = name.split("_")      # search_<prior>_<rollout>
        path = ("results/student_prior.npz" if roll == "j"
                else "results/student_laya.npz")
        return HybridSearch(path=path, rollouts=rollouts,
                            prior_model=prior_model,
                            rollout_label=roll.upper())
    if name == "jev_search":
        return HybridSearch(rollouts=rollouts)
    if name == "laya_search":
        return HybridSearch(rollouts=rollouts, prior_model="laya")
    if name == "jev_or":
        from sysone.core import JevBackend
        return JevOnly(JevBackend(via="openrouter"), label="JEV(or)")
    if name == "laya":
        from sysone.core import LayaBackend
        return JevOnly(LayaBackend(), compact=True, label="LAYA")
    if name == "laya_tuned":
        return LayaPolicy()
    if name == "jev_compact":
        return JevOnly(backend, compact=True, label="JEV(compact)")
    if name == "bestof":
        return JevBestOfN(backend, rollouts=rollouts)
    if name == "jev_mcts":
        return MCTSPolicy(backend, iterations, use_jev=True)
    if name == "uniform_mcts":
        return MCTSPolicy(None, iterations, use_jev=False)
    raise ValueError(f"unknown policy: {name}")


CONFIGS = ["jev", "bestof", "jev_mcts", "uniform_mcts",
           "student_prior", "student_improved"]
