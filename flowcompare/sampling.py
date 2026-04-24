"""Process-agnostic generative sampling.

The whole point of the :class:`SequenceProcess` abstraction is that sampling
is the same program for every process: take the prior, iterate
``network -> process.step`` for ``n_steps``, decode the final state to
tokens. This module provides that single program.

Two concrete entry points:

- :func:`sample_tokens` -- returns an ``int32`` array of shape ``(B, L)``.
  Pads are preserved: if a ``mask`` is supplied, ``PAD_ID`` fills the masked
  positions of the final token tensor. The caller chooses the length ``L``
  and the batch size.
- :func:`sample_strings` -- thin wrapper that decodes via a
  :class:`ProteinTokenizer`.

NFE (Number of Function Evaluations) here is exactly the number of network
forward passes, which equals ``n_steps``. The eval suite's Pareto curve
sweeps this value.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np

from flowcompare.processes.base import ProcessState, SequenceProcess
from flowcompare.tokenizer import ProteinTokenizer


class _ModelApply(Protocol):
    def __call__(self, params: Any, x: jnp.ndarray, t: jnp.ndarray, mask: jnp.ndarray | None) -> jnp.ndarray: ...


@dataclass
class SamplingTrace:
    """What every ``n_steps`` sampling run records.

    ``tokens_per_step`` is a list of ``(B, L)`` int32 arrays capturing the
    decoded token sequence at each step; ``t_per_step`` records the scalar
    time at each step. Used by the Pareto-curve eval and by the theory-doc
    figures. For production sampling set ``record=False`` to skip.
    """

    final_tokens: np.ndarray
    tokens_per_step: list[np.ndarray]
    t_per_step: list[np.ndarray]


def decode_state_to_tokens(
    state: ProcessState, process: SequenceProcess
) -> np.ndarray:
    """Recover ``(B, L)`` int32 tokens from a process state.

    - If ``state.extras['tokens']`` is set (DFM), use it directly.
    - Otherwise argmax along the last axis of ``state.tensor``.
    """
    if state.extras is not None and "tokens" in state.extras:
        return np.asarray(state.extras["tokens"], dtype=np.int32)
    return np.asarray(state.tensor).argmax(axis=-1).astype(np.int32)


def sample_tokens(
    process: SequenceProcess,
    model_apply: _ModelApply,
    params: Any,
    *,
    rng: np.random.Generator,
    batch_size: int,
    length: int,
    n_steps: int,
    mask: np.ndarray | None = None,
    pad_id: int = 0,
    record: bool = False,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> SamplingTrace:
    """Generate token sequences by running the process's CTMC backwards in t.

    Parameters
    ----------
    process :
        Instance of :class:`SequenceProcess` whose ``sample_prior`` / ``step``
        define the generative dynamics.
    model_apply :
        Typically ``model.apply`` from a Flax module. Must accept
        ``(params, x, t, mask)`` and return logits of shape
        ``(batch_size, length, vocab)``.
    params :
        Parameter tree for the model.
    rng :
        Seeded numpy generator; drives all stochasticity (prior, step noise,
        categorical samples). The same seed therefore reproduces the same
        sample.
    batch_size, length :
        Desired output shape.
    n_steps :
        Number of discretisation / NFE steps.
    mask :
        Optional bool mask of shape ``(batch_size, length)`` marking which
        positions are "real". Padding positions are filled with ``pad_id``
        at the end.
    pad_id :
        Integer inserted at masked positions in the final output.
    record :
        If True, capture per-step token snapshots.
    t_start, t_end :
        Time window; standard sampling uses ``[0, 1]``.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}.")
    state = process.sample_prior(rng, (batch_size, length), mask=mask)
    dt = (t_end - t_start) / n_steps
    tokens_trace: list[np.ndarray] = []
    t_trace: list[np.ndarray] = []
    for _ in range(n_steps):
        x = jnp.asarray(state.tensor)
        t = jnp.asarray(state.t)
        m = jnp.asarray(mask) if mask is not None else None
        logits = np.asarray(model_apply(params, x, t, m))
        state = process.step(rng, logits, state, dt)
        if record:
            tokens_trace.append(decode_state_to_tokens(state, process))
            t_trace.append(np.asarray(state.t))
    final_tokens = decode_state_to_tokens(state, process)
    if mask is not None:
        final_tokens = np.where(mask, final_tokens, np.int32(pad_id)).astype(np.int32)
    return SamplingTrace(
        final_tokens=final_tokens,
        tokens_per_step=tokens_trace,
        t_per_step=t_trace,
    )


def sample_strings(
    process: SequenceProcess,
    model_apply: _ModelApply,
    params: Any,
    tokenizer: ProteinTokenizer,
    *,
    rng: np.random.Generator,
    batch_size: int,
    length: int,
    n_steps: int,
    mask: np.ndarray | None = None,
    strip_special: bool = True,
) -> list[str]:
    """Convenience wrapper: sample tokens and decode to amino-acid strings."""
    trace = sample_tokens(
        process,
        model_apply,
        params,
        rng=rng,
        batch_size=batch_size,
        length=length,
        n_steps=n_steps,
        mask=mask,
        pad_id=tokenizer.PAD_ID,
    )
    return [
        tokenizer.decode(row, skip_special=strip_special)
        for row in trace.final_tokens
    ]


def nfe_sweep(
    process: SequenceProcess,
    model_apply: _ModelApply,
    params: Any,
    *,
    rng_factory: Callable[[], np.random.Generator],
    batch_size: int,
    length: int,
    n_steps_list: list[int],
    mask: np.ndarray | None = None,
    pad_id: int = 0,
) -> dict[int, np.ndarray]:
    """Generate samples at each NFE in ``n_steps_list`` from the same prior.

    For a fair Pareto comparison, every point in the sweep starts from the
    same ``rng`` state; ``rng_factory`` is called fresh per NFE value so that
    downstream randomness does not drift across points.
    """
    out: dict[int, np.ndarray] = {}
    for n in n_steps_list:
        trace = sample_tokens(
            process,
            model_apply,
            params,
            rng=rng_factory(),
            batch_size=batch_size,
            length=length,
            n_steps=n,
            mask=mask,
            pad_id=pad_id,
        )
        out[int(n)] = trace.final_tokens
    return out
