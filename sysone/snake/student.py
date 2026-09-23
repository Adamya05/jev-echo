"""The MLP student (phase 2), in MLX.

Distillation, not RL. The teacher hands over a full distribution per state
and the student matches it with KL. Soft targets rather than argmax labels
is the whole economy of the thing: an argmax label carries log2(4) = 2 bits,
while a distribution carries the teacher's whole relative ranking, which is
most of what makes distillation cheaper than learning from scratch.

Two things measured earlier shape the training here:

  * ~91% of states have a snake shorter than 8 cells, where the moves are
    nearly indistinguishable (feature spread 0.021 against 0.088 for long
    snakes). Training uniformly means spending 91% of gradient steps on
    states where the answer barely matters.
  * The fix is to weight by how much opinion the teacher actually has.
    Low-entropy teacher distributions are the informative ones.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from sysone.core import Answer, Decision
from sysone.snake import features as F
from sysone.snake.game import Snake

NEG = -1e9


class StudentNet(nn.Module):
    """28 -> hidden -> hidden -> 4. Deliberately tiny."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.l1 = nn.Linear(F.N_FEATURES, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, len(F.MOVES))

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        return self.l3(x)


def masked_log_softmax(logits: mx.array, mask: mx.array) -> mx.array:
    """Unavailable moves are removed before normalizing, not after.

    Masking after the softmax would leave probability mass stranded on
    moves that do not exist and quietly mis-normalize everything else.
    """
    return nn.log_softmax(logits + mx.where(mask > 0, 0.0, NEG), axis=-1)


def kl_loss(model: StudentNet, x: mx.array, mask: mx.array,
            target: mx.array, weight: mx.array) -> mx.array:
    logp = masked_log_softmax(model(x), mask)
    safe = mx.where(target > 0, target, 1.0)          # keep log finite
    per_state = mx.sum(
        mx.where(target > 0, target * (mx.log(safe) - logp), 0.0), axis=-1)
    return mx.sum(per_state * weight) / mx.sum(weight)


def entropy_weights(target: mx.array, floor: float = 0.05) -> mx.array:
    """Weight each state by how decisive the teacher was.

    A uniform teacher distribution says "these moves are interchangeable",
    which is true and also worth almost nothing to learn from. A peaked one
    is a real preference. Normalized so a uniform state keeps `floor`
    weight rather than none -- they still teach the mask.
    """
    safe = mx.where(target > 0, target, 1.0)
    h = -mx.sum(mx.where(target > 0, target * mx.log(safe), 0.0), axis=-1)
    n_avail = mx.maximum(mx.sum(target > 0, axis=-1), 1)
    h_max = mx.log(n_avail.astype(mx.float32))
    decisiveness = 1.0 - h / mx.maximum(h_max, 1e-6)
    return floor + (1.0 - floor) * decisiveness


def train(data: Mapping[str, Any], *, hidden: int = 64, epochs: int = 60,
          batch: int = 256, lr: float = 3e-3, weighted: bool = True,
          seed: int = 0, verbose: bool = True) -> tuple[StudentNet, dict]:
    mx.random.seed(seed)
    X = mx.array(data["features"], dtype=mx.float32)
    M = mx.array(data["mask"], dtype=mx.float32)
    T = mx.array(data["target"], dtype=mx.float32)
    W = entropy_weights(T) if weighted else mx.ones(T.shape[0])

    n = X.shape[0]
    cut = int(n * 0.9)
    idx = mx.array(list(range(n)))
    model = StudentNet(hidden)
    opt = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, kl_loss)

    history = {"train": [], "val": []}
    for epoch in range(epochs):
        perm = mx.random.permutation(idx[:cut])
        total, seen = 0.0, 0
        for i in range(0, cut, batch):
            b = perm[i:i + batch]
            loss, grads = loss_and_grad(model, X[b], M[b], T[b], W[b])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            total += float(loss) * len(b)
            seen += len(b)
        v = float(kl_loss(model, X[cut:], M[cut:], T[cut:], W[cut:]))
        history["train"].append(total / max(seen, 1))
        history["val"].append(v)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:>3}  train KL {total/max(seen,1):.4f}  val KL {v:.4f}")

    n_params = sum(p.size for _, p in _flat(model.parameters()))
    return model, {"history": history, "params": n_params, "n_states": n}


def _flat(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flat(v, f"{prefix}{k}.")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flat(v, f"{prefix}{i}.")
    else:
        yield prefix, tree


def save(model: StudentNet, path: str | Path) -> None:
    flat = {k.rstrip("."): v for k, v in _flat(model.parameters())}
    mx.savez(str(path), **flat)


def load(path: str | Path, hidden: int = 64) -> StudentNet:
    model = StudentNet(hidden)
    model.load_weights(str(path))
    mx.eval(model.parameters())
    return model


# --------------------------------------------------------------------------
# The same decide() seam the hosted backend uses
# --------------------------------------------------------------------------

class StudentBackend:
    """A local specialist on the shared seam.

    Free, offline, and roughly four orders of magnitude faster than the
    hosted call -- but it answers exactly one question about exactly one
    game. That is the trade the three-tier story is about.
    """

    def __init__(self, model: StudentNet | str | Path, hidden: int = 64) -> None:
        self.model = load(model, hidden) if isinstance(model, (str, Path)) else model
        self.name = "student:mlp"

    def predict(self, game: Snake) -> dict[str, float]:
        facts = game.survivable_moves()
        x = mx.array([F.encode(game, facts)], dtype=mx.float32)
        m = mx.array([F.mask(facts)], dtype=mx.float32)
        p = mx.exp(masked_log_softmax(self.model(x), m))
        mx.eval(p)
        return {mv: float(p[0, i]) for i, mv in enumerate(F.MOVES)
                if facts.get(mv) is not None}

    def decide(self, state: Any, questions: Mapping[str, Any]) -> Decision:
        game = state if isinstance(state, Snake) else state["game"]
        t0 = time.perf_counter()
        probs = self.predict(game)
        ms = (time.perf_counter() - t0) * 1000
        best = max(probs, key=probs.get) if probs else game.direction
        ordered = sorted(probs.values(), reverse=True)
        conf = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        key = next(iter(questions), "move")
        return Decision(
            answers={key: Answer("choice", best, conf, probs)},
            latency_ms=ms, input_tokens=0, model="student-mlp",
        )
