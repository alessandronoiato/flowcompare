"""Time-samplers for continuous-time loss estimators.

Both BFN and DFM define their training objective as an expectation over
``t ~ U[0, 1]``. A naive ``uniform`` sampler works, but low-discrepancy
(Latin hypercube / stratified) sampling reduces estimator variance for small
batches and is a standard trick in continuous-time diffusion / flow training.

All samplers operate on a ``numpy.random.Generator`` so they remain callable
outside the jitted trainer boundary.
"""

from __future__ import annotations

import numpy as np


def sample_time(
    rng: np.random.Generator,
    batch_size: int,
    *,
    scheme: str = "uniform",
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> np.ndarray:
    """Draw ``batch_size`` per-example times in ``[t_min, t_max]``.

    Parameters
    ----------
    rng :
        Seeded numpy generator.
    batch_size :
        Number of samples to draw.
    scheme :
        ``"uniform"`` for IID uniform samples; ``"stratified"`` partitions
        ``[t_min, t_max]`` into ``batch_size`` equal bins and draws one point
        per bin (Latin hypercube / variance-reduction).
    t_min, t_max :
        Lower and upper truncation. DFM training often uses ``t_max = 1 - eps``
        to avoid the ``1/(1-t)`` singularity; BFN can use the full range.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if not (0.0 <= t_min < t_max <= 1.0):
        raise ValueError(
            f"need 0 <= t_min < t_max <= 1, got t_min={t_min}, t_max={t_max}."
        )
    if scheme == "uniform":
        u = rng.uniform(0.0, 1.0, size=(batch_size,)).astype(np.float32)
    elif scheme == "stratified":
        edges = np.linspace(0.0, 1.0, batch_size + 1, dtype=np.float32)
        within = rng.uniform(0.0, 1.0, size=(batch_size,)).astype(np.float32)
        u = edges[:-1] + within * (edges[1:] - edges[:-1])
        rng.shuffle(u)
    else:
        raise ValueError(f"unknown scheme {scheme!r}.")
    return (t_min + u * (t_max - t_min)).astype(np.float32)
