"""Train Echo: a small network that copies Jev's answers on raw boards.

    uv run python collect_raw.py --games 90     # Jev labels ~13.7k boards
    uv run python train_board.py                # ~30 s on an M4
"""
import time
import mlx.core as mx
import numpy as np
from sysone.snake import boardnet as B

mx.set_default_device(mx.cpu)

d = np.load("results/dataset_board.npz")
X = B.encode_many(d["boards"])
M, T = d["mask"], d["target"]

t0 = time.perf_counter()
model, hist = B.train(X, M, T, epochs=90)
secs = time.perf_counter() - t0

pred = np.array(mx.argmax(B.masked_log_softmax(model(mx.array(X)), mx.array(M)), -1))
agree = (pred == T.argmax(1)).mean()
B.save(model, "results/student_board.npz")

print(f"{len(X):,} boards  trained in {secs:.0f}s  agrees with Jev on {agree:.1%}")
print(f"{B.n_params(model):,} parameters -> results/student_board.npz")
