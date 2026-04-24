"""On-the-fly random protein sequences for smoketests and CI.

The generator draws amino acids from a fixed distribution (optionally weighted
to roughly match natural protein frequencies) and sequence lengths uniformly
from a given range. Given a seeded numpy ``Generator``, output is perfectly
reproducible.
"""

from __future__ import annotations

import numpy as np

from flowcompare.tokenizer import STANDARD_AMINO_ACIDS

# Approximate natural amino-acid frequencies in UniProt/Swiss-Prot, rounded
# into percentages for the same ACDEFGHIKLMNPQRSTVWY order as the tokenizer.
# Useful when smoketest evals want to see that a trained model learned
# *something* non-uniform; exact values are not critical.
_NATURAL_FREQS = np.array(
    [
        0.0825, 0.0139, 0.0546, 0.0672, 0.0386, 0.0708, 0.0227, 0.0591,
        0.0582, 0.0965, 0.0242, 0.0406, 0.0473, 0.0393, 0.0553, 0.0660,
        0.0535, 0.0686, 0.0109, 0.0292,
    ],
    dtype=np.float64,
)
_NATURAL_FREQS = _NATURAL_FREQS / _NATURAL_FREQS.sum()


def generate_synthetic_proteins(
    rng: np.random.Generator,
    n_sequences: int,
    *,
    min_length: int = 30,
    max_length: int = 80,
    distribution: str = "uniform",
) -> list[str]:
    """Return a list of ``n_sequences`` synthetic protein strings.

    Parameters
    ----------
    rng :
        Seeded numpy generator.
    n_sequences :
        Number of sequences to produce.
    min_length, max_length :
        Inclusive bounds on sequence length.
    distribution :
        ``"uniform"`` for uniform amino-acid distribution, ``"natural"`` for
        Swiss-Prot-ish frequencies.
    """
    if n_sequences <= 0:
        raise ValueError(f"n_sequences must be positive, got {n_sequences}.")
    if min_length <= 0 or max_length < min_length:
        raise ValueError(
            f"invalid length bounds: min_length={min_length}, max_length={max_length}."
        )
    if distribution == "uniform":
        probs = None
    elif distribution == "natural":
        probs = _NATURAL_FREQS
    else:
        raise ValueError(f"unknown distribution {distribution!r}.")

    alphabet = np.array(list(STANDARD_AMINO_ACIDS))
    seqs: list[str] = []
    for _ in range(n_sequences):
        length = int(rng.integers(min_length, max_length + 1))
        idx = rng.choice(len(alphabet), size=length, p=probs)
        seqs.append("".join(alphabet[idx].tolist()))
    return seqs
