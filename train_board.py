"""Train Echo: a small network that copies Jev's answers on raw boards.

    uv run python collect_raw.py --games 90     # Jev labels ~13.7k boards
    uv run python train_board.py                # ~30 s on an M4
"""
import argparse, time
import mlx.core as mx
import numpy as np
from sysone.snake import boardnet as B

mx.set_default_device(mx.cpu)

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", default=["results/dataset_board.npz"])
ap.add_argument("--out", default="results/student_board.npz")
a = ap.parse_args()
ds = [np.load(p) for p in a.data]
X = B.encode_many(np.concatenate([d["boards"] for d in ds]))
M = np.concatenate([d["mask"] for d in ds])
T = np.concatenate([d["target"] for d in ds])

t0 = time.perf_counter()
model, hist = B.train(X, M, T, epochs=90)
secs = time.perf_counter() - t0

pred = np.array(mx.argmax(B.masked_log_softmax(model(mx.array(X)), mx.array(M)), -1))
agree = (pred == T.argmax(1)).mean()
B.save(model, a.out)

print(f"{len(X):,} boards  trained in {secs:.0f}s  agrees with Jev on {agree:.1%}")
print(f"{B.n_params(model):,} parameters -> {a.out}")
