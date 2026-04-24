"""NFE-vs-quality Pareto sweep.

The headline figure in the theory doc: for a fixed trained model, sweep the
number of sampling steps ``n_steps`` (== number of function evaluations),
and at each point compute one or more downstream metrics (diversity, ESM
pseudo-likelihood, ESMFold quality). The interface here just handles the
sampling half; the caller plugs in its own metric callable.

This is the main place where BFN and DFM diverge visually: BFN typically
requires many more NFE to hit the same quality because its updates are
low-rate Gaussian observations, whereas mask-interpolant DFM jumps discrete
positions in chunks. The theory doc expects this prediction and backs it
out of the math.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from flowcompare.processes.base import SequenceProcess
from flowcompare.sampling import sample_tokens
from flowcompare.tokenizer import ProteinTokenizer


def pareto_sweep(
    process: SequenceProcess,
    model,
    params: Any,
    tokenizer: ProteinTokenizer,
    *,
    rng: np.random.Generator,
    batch_size: int,
    length: int,
    nfe_values: Iterable[int],
    metrics: dict[str, Callable[[list[str]], float]],
    n_batches: int = 1,
) -> list[dict[str, float]]:
    """Sample at each NFE and report the named metrics.

    Parameters
    ----------
    process, model, params, tokenizer :
        As for :func:`flowcompare.sampling.sample_strings`.
    rng :
        Shared RNG. We call ``np.random.default_rng(rng.integers(...))`` per
        point so that the samples at each NFE differ but are reproducible
        given a seed.
    batch_size, length :
        Output shape per sample call.
    nfe_values :
        Iterable of ``n_steps`` to evaluate at.
    metrics :
        Mapping name -> callable ``list[str] -> float``. Typical entries:
        ``{"mean_identity": eval.diversity.mean_pairwise_identity}``.
    n_batches :
        Number of sample-batches to pool per NFE. More stabilises metrics at
        cost of sampling time.
    """
    results: list[dict[str, float]] = []
    for n in nfe_values:
        samples: list[str] = []
        for _ in range(n_batches):
            seed = int(rng.integers(0, 2**31 - 1))
            trace = sample_tokens(
                process,
                model.apply,
                params,
                rng=np.random.default_rng(seed),
                batch_size=batch_size,
                length=length,
                n_steps=int(n),
            )
            samples.extend(
                tokenizer.decode(row, skip_special=True) for row in trace.final_tokens
            )
        row: dict[str, float] = {"nfe": float(n), "n_samples": float(len(samples))}
        for name, fn in metrics.items():
            row[name] = float(fn(samples))
        results.append(row)
    return results
