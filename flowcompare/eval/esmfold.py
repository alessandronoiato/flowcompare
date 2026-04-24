"""ESMFold foldability evaluation via a torch subprocess.

Same subprocess pattern as :mod:`flowcompare.eval.esm`. ESMFold is much
heavier (several GB of weights) so we gate availability behind an explicit
install marker.

Outputs of interest for the benchmark:
- Per-sample pLDDT (confidence of predicted structure), averaged across
  residues.
- Per-sample pTM (estimate of TM-score against ground truth).
High pLDDT means the model produces sequences that ESMFold believes fold into
confident structures -- the best single-number proxy for "is this a plausible
protein" that does not require wet-lab validation.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence


def is_available() -> bool:
    return importlib.util.find_spec("esm") is not None and importlib.util.find_spec(
        "torch"
    ) is not None


_INSTALL_HINT = (
    "ESMFold foldability evaluation requires torch and fair-esm and a machine "
    "with >= 24 GB GPU memory. Install with:  pip install torch fair-esm  "
    "(the eval is run in a separate subprocess)."
)


def compute_foldability(
    sequences: Sequence[str],
    *,
    device: str = "cuda",
    chunk_size: int = 64,
) -> list[dict[str, float]]:
    """Return ``[{plddt, ptm, len}]`` per sequence.

    Raises :class:`RuntimeError` if ESMFold is not installed; the subprocess
    worker is deferred pending the first benchmark run.
    """
    if not is_available():
        raise RuntimeError(_INSTALL_HINT)
    raise NotImplementedError(
        "ESMFold worker subprocess not yet implemented; interface is stable. "
        "See flowcompare/eval/esmfold.py docstring for the planned path."
    )
