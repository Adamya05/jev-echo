"""A student that reads the same raw board Jev does.

The feature-based student was not a like-for-like comparison: it consumed 28
numbers that code had already worked out, while Jev read an ASCII grid. This
one takes the grid.

Input: 10x10x3 planes (body, head, food). No flood fill, no distances, no
derived facts of any kind -- the same information Jev gets, in the same
spatial form.

A small conv net rather than an MLP, because the task is spatial and a
convolution is the right prior for a grid. Still small enough to sit on a
microcontroller-class device.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from sysone.core import Answer, Decision
from sysone.snake.game import Snake

MOVES = ("UP", "DOWN", "LEFT", "RIGHT")
NEG = -1e9


def encode_board(flat: str) -> np.ndarray:
    """'....o@...*' (100 chars) -> 10x10x3 float planes."""
    a = np.frombuffer(flat.encode(), dtype=np.uint8)
    g = a.reshape(10, 10)
    return np.stack([(g == ord("o")), (g == ord("@")), (g == ord("*"))],
                    axis=-1).astype(np.float32)


def encode_many(boards) -> np.ndarray:
    return np.stack([encode_board(b) for b in boards])


class BoardNet(nn.Module):
    """3 planes -> conv -> conv -> pool -> dense -> 4 moves."""

    def __init__(self, ch: int = 16, hidden: int = 48) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.fc1 = nn.Linear(ch * 25, hidden)
        self.fc2 = nn.Linear(hidden, len(MOVES))

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.c1(x))
        x = nn.relu(self.c2(x))
        b, h, w, c = x.shape                      # 10x10 -> 5x5 average pool
        x = x.reshape(b, h // 2, 2, w // 2, 2, c).mean(axis=(2, 4))
        x = x.reshape(b, -1)
        return self.fc2(nn.relu(self.fc1(x)))


def masked_log_softmax(logits: mx.array, mask: mx.array) -> mx.array:
    return nn.log_softmax(logits + mx.where(mask > 0, 0.0, NEG), axis=-1)


def kl(model, x, mask, target, w):
    logp = masked_log_softmax(model(x), mask)
    safe = mx.where(target > 0, target, 1.0)
    per = mx.sum(mx.where(target > 0, target * (mx.log(safe) - logp), 0.0), axis=-1)
    return mx.sum(per * w) / mx.sum(w)


def entropy_weights(t: mx.array, floor: float = 0.05) -> mx.array:
    safe = mx.where(t > 0, t, 1.0)
    h = -mx.sum(mx.where(t > 0, t * mx.log(safe), 0.0), axis=-1)
    n = mx.maximum(mx.sum(t > 0, axis=-1), 1).astype(mx.float32)
    return floor + (1 - floor) * (1.0 - h / mx.maximum(mx.log(n), 1e-6))


def train(X, M, T, *, epochs=90, batch=256, lr=2e-3, ch=16, hidden=48, seed=0):
    mx.random.seed(seed)
    X, M, T = mx.array(X), mx.array(M), mx.array(T)
    W = entropy_weights(T)
    n = X.shape[0]; cut = int(n * 0.9)
    model = BoardNet(ch, hidden)
    opt = optim.Adam(learning_rate=lr)
    step = nn.value_and_grad(model, kl)
    hist = []
    for e in range(epochs):
        perm = mx.random.permutation(mx.arange(cut))
        for i in range(0, cut, batch):
            b = perm[i:i + batch]
            loss, grads = step(model, X[b], M[b], T[b], W[b])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
        hist.append(float(kl(model, X[cut:], M[cut:], T[cut:], W[cut:])))
    return model, hist


def _flat(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items(): yield from _flat(v, f"{prefix}{k}.")
    elif isinstance(tree, list):
        for i, v in enumerate(tree): yield from _flat(v, f"{prefix}{i}.")
    else:
        yield prefix.rstrip("."), tree


def n_params(model) -> int:
    return sum(p.size for _, p in _flat(model.parameters()))


def save(model, path): mx.savez(str(path), **dict(_flat(model.parameters())))


def load(path, ch=16, hidden=48):
    m = BoardNet(ch, hidden); m.load_weights(str(path)); mx.eval(m.parameters()); return m


class BoardBackend:
    """Same seam, same input as Jev."""

    def __init__(self, model, ch: int = 16, hidden: int = 48) -> None:
        mx.set_default_device(mx.cpu)
        self.model = load(model, ch, hidden) if isinstance(model, (str, Path)) else model
        self.name = "student:board"

    def predict(self, game: Snake) -> dict[str, float]:
        facts = game.survivable_moves()
        x = mx.array(encode_many([game.ascii_board().replace("\n", "")]))
        m = mx.array([[1.0 if mv in facts else 0.0 for mv in MOVES]])
        p = mx.exp(masked_log_softmax(self.model(x), m))
        mx.eval(p)
        return {mv: float(p[0, i]) for i, mv in enumerate(MOVES) if mv in facts}

    def decide(self, state: Any, questions: Mapping[str, Any]) -> Decision:
        game = state if isinstance(state, Snake) else state["game"]
        t0 = time.perf_counter()
        probs = self.predict(game)
        ms = (time.perf_counter() - t0) * 1000
        best = max(probs, key=probs.get) if probs else game.direction
        o = sorted(probs.values(), reverse=True)
        conf = o[0] - (o[1] if len(o) > 1 else 0.0)
        return Decision(answers={next(iter(questions), "move"):
                                 Answer("choice", best, conf, probs)},
                        latency_ms=ms, input_tokens=0, model="board-cnn",
                        billed=False)
