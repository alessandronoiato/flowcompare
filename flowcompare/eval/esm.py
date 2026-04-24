"""ESM-2 pseudo-likelihood evaluation via a torch subprocess.

ESM-2 ships only in PyTorch; rather than pull torch into the core package
we shell out to a dedicated script when the user wants these numbers. The
interface here:

- :func:`is_available` -- probes whether ``torch`` and ``fair-esm`` are
  importable in the active environment. Pure check, no side effects.
- :func:`compute_pseudo_likelihood` -- runs the evaluation. If torch/esm are
  missing it raises ``RuntimeError`` with a clear install pointer.

Implementation note: the actual subprocess path writes sequences to a temp
FASTA, invokes ``python -m flowcompare.eval._esm_worker`` (not yet
implemented), and reads back a JSON of per-sequence PLLs. This module exposes
the interface and the availability check; the worker will land when the
first real benchmark run is configured.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence


def is_available() -> bool:
    """True iff ``torch`` and ``esm`` (fair-esm) import in the current env."""
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("esm") is not None
    )


_INSTALL_HINT = (
    "ESM-2 pseudo-likelihood evaluation requires torch and fair-esm. "
    "Install with:  pip install torch fair-esm  "
    "(the eval is run in a separate subprocess so it does not pollute the "
    "JAX environment)."
)


def compute_pseudo_likelihood(
    sequences: Sequence[str],
    *,
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 8,
    device: str = "cpu",
) -> list[float]:
    """Return per-sequence ESM-2 pseudo-likelihood (higher = more protein-like).

    Always raises :class:`RuntimeError` if ESM / torch are not installed; the
    worker implementation is forthcoming. Call :func:`is_available` first if
    you want to branch on availability.
    """
    if not is_available():
        raise RuntimeError(_INSTALL_HINT)
    # The worker subprocess is intentionally deferred: we want the interface
    # stable and tested before the first benchmark runs. Tracked in the
    # repo TODOs.
    raise NotImplementedError(
        "ESM-2 worker subprocess not yet implemented; interface is stable. "
        "See flowcompare/eval/esm.py docstring for the planned path."
    )
