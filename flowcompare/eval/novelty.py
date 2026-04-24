"""Novelty metric: how close are generated samples to the training corpus?

For each sample we compute the maximum position-wise identity against any
training sequence (shared-length prefix comparison, matching the diversity
module's convention). High max-identity means the model memorised something
close to a training sequence; low max-identity means the samples look novel.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i]) / n


def max_identity_to_train(
    samples: Sequence[str], train: Sequence[str]
) -> np.ndarray:
    """Return an array of per-sample maximum identity against any training sequence."""
    if len(train) == 0:
        raise ValueError("train corpus is empty.")
    out = np.zeros(len(samples), dtype=np.float32)
    for i, s in enumerate(samples):
        best = 0.0
        for t in train:
            ident = _identity(s, t)
            if ident > best:
                best = ident
                if best == 1.0:
                    break
        out[i] = best
    return out
