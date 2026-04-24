"""Diversity metrics on a set of generated sequences.

Pairwise sequence identity is the fraction of aligned positions that match,
evaluated position-wise over equal-length pairs (we do not run a full
alignment -- that would be expensive and is unnecessary for fixed-length
samples). For variable-length samples the shorter is implicitly compared
only at its actual positions and the identity is normalised by the shorter
length. This matches the convention used in Gat et al. 2024.

Returned values are in ``[0, 1]``; lower is more diverse.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


def pairwise_identity(sequences: Sequence[str]) -> np.ndarray:
    """Return an ``(n, n)`` matrix of pairwise identities with 1.0 on the diagonal."""
    n = len(sequences)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            ident = _identity(sequences[i], sequences[j])
            out[i, j] = ident
            out[j, i] = ident
    return out


def mean_pairwise_identity(sequences: Sequence[str]) -> float:
    """Mean off-diagonal pairwise identity -- the headline diversity number.

    Lower is more diverse. Random protein sequences over the 20-AA alphabet
    score near ``1/20 = 0.05``; natural protein families often score in the
    0.2 - 0.5 range depending on family conservation.
    """
    n = len(sequences)
    if n < 2:
        raise ValueError(f"need at least 2 sequences, got {n}.")
    mat = pairwise_identity(sequences)
    off_diag_sum = mat.sum() - np.trace(mat)
    return float(off_diag_sum / (n * (n - 1)))
