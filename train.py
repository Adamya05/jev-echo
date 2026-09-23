"""Train the students and ablate the entropy weighting.

    uv run python train.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from sysone.snake import student as S

mx.set_default_device(mx.cpu)   # 6k params: CPU beats GPU ~10x at this size

d = np.load("results/dataset.npz")
X, M = d["features"], d["mask"]
print(f"{len(X):,} states   snake length median {np.median(d['length']):.0f}, "
      f"p90 {np.percentile(d['length'], 90):.0f}\n")

results = {}
for target_name in ("prior", "improved"):
    T = d[target_name]
    for weighted in (True, False):
        tag = f"{target_name}{'' if weighted else ' (unweighted)'}"
        t0 = time.perf_counter()
        model, meta = S.train({"features": X, "mask": M, "target": T},
                              epochs=80, weighted=weighted, verbose=False)
        secs = time.perf_counter() - t0

        logits = S.masked_log_softmax(model(mx.array(X)), mx.array(M))
        agree = float((np.array(mx.argmax(logits, -1)) == T.argmax(1)).mean())
        # Agreement on the states that actually carry an opinion.
        w = np.array(S.entropy_weights(mx.array(T)))
        sharp = w > np.percentile(w, 75)
        agree_sharp = float((np.array(mx.argmax(logits, -1))[sharp]
                             == T.argmax(1)[sharp]).mean())

        results[tag] = (model, meta, agree, agree_sharp)
        print(f"{tag:<24} val KL {meta['history']['val'][-1]:.4f}   "
              f"argmax {agree:.1%}   on decisive states {agree_sharp:.1%}   "
              f"{secs:.1f}s")

print()
for name in ("prior", "improved"):
    model = results[name][0]
    S.save(model, f"results/student_{name}.npz")
    print(f"saved results/student_{name}.npz  ({results[name][1]['params']:,} params)")

# How often do the two students actually disagree with each other?
lp = S.masked_log_softmax(results["prior"][0](mx.array(X)), mx.array(M))
li = S.masked_log_softmax(results["improved"][0](mx.array(X)), mx.array(M))
disagree = float((np.array(mx.argmax(lp, -1)) != np.array(mx.argmax(li, -1))).mean())
print(f"\nthe two students disagree on {disagree:.1%} of states "
      f"(their teachers disagreed on 7.7%)")
