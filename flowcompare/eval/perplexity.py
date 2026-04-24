"""Held-out loss / perplexity estimator.

Both BFN and DFM define a continuous-time bound on negative log-likelihood.
Here we evaluate that bound on a held-out set by averaging the training loss
over many ``t`` samples per sequence. More samples = lower variance.

The number returned is per-token loss in nats (smaller is better). For BFN
and DFM this is a stochastic upper bound on ``-log p(x)``; perplexity can be
computed by the caller as ``exp(loss)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp
import numpy as np

from flowcompare.processes.base import SequenceProcess
from flowcompare.training.schedules import sample_time
from flowcompare.training.trainer import make_eval_step


def compute_held_out_loss(
    model,
    params: Any,
    process: SequenceProcess,
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    rng: np.random.Generator,
    n_time_samples: int = 4,
    time_scheme: str = "uniform",
    t_max: float = 1.0,
) -> float:
    """Estimate the per-token continuous-time loss on ``batches``.

    Parameters
    ----------
    model :
        Flax module whose ``apply`` was used for training.
    params :
        Parameter tree (typically ``state.params`` or ``state.ema_params``).
    process :
        The generative process used during training.
    batches :
        Iterable of ``(ids, mask)`` batches. Usually produced by the same
        ``iterate_batches`` as training, over a held-out dataset.
    rng :
        Seeded numpy generator; advances deterministically across the call.
    n_time_samples :
        Number of independent ``t`` draws per batch. Averaging shrinks
        variance as ``1/n_time_samples``.
    time_scheme, t_max :
        Forwarded to :func:`sample_time`.
    """
    if n_time_samples <= 0:
        raise ValueError(f"n_time_samples must be positive, got {n_time_samples}.")
    eval_step = make_eval_step(model, process)
    total_loss = 0.0
    total_tokens = 0
    for ids, mask in batches:
        n_real = int(mask.sum())
        if n_real == 0:
            continue
        batch_loss = 0.0
        for _ in range(n_time_samples):
            t = sample_time(rng, ids.shape[0], scheme=time_scheme, t_max=t_max)
            state = process.corrupt(rng, ids, t, mask=mask)
            loss = float(
                eval_step(
                    params,
                    jnp.asarray(state.tensor),
                    jnp.asarray(state.t),
                    jnp.asarray(ids),
                    jnp.asarray(mask),
                )
            )
            batch_loss += loss
        batch_loss /= n_time_samples
        total_loss += batch_loss * n_real
        total_tokens += n_real
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens
