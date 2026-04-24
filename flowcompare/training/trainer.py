"""Jitted training and evaluation loops.

Design:

- The process is deliberately kept in numpy-land because its sampling logic
  depends on ``np.random.Generator`` and is awkward to jit. Per step we:

  1. Sample ``(ids, mask)`` from the dataset (numpy).
  2. Sample ``t`` via :func:`~flowcompare.training.schedules.sample_time`.
  3. Call ``process.corrupt(rng, ids, t, mask=mask)`` to get the state.
  4. Push ``(x_t, t, x1, mask)`` through the jitted ``train_step``.

- Inside ``train_step`` we call ``process.jax_loss_fn()`` (a pure-JAX
  closure) and take its gradient with respect to the network parameters.
  Parameters + optimizer state + a slow EMA copy live in :class:`TrainState`.
- Loss aggregation normalises by the number of real tokens in the batch so
  that batches of varying padding-density are comparable.

Shapes:
- ``ids``: ``(B, L)`` int32
- ``mask``: ``(B, L)`` bool
- ``x_t``: ``(B, L, V)`` float32 (one-hot or simplex, process-specific)
- ``t``: ``(B,)`` float32
- ``logits``: ``(B, L, V)`` float32
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flowcompare.processes.base import SequenceProcess
from flowcompare.training.schedules import sample_time


class TrainState(flax.struct.PyTreeNode):
    """Minimal training state, registered as a pytree so jit can traverse it.

    EMA params are updated with ``ema = decay * ema + (1 - decay) * params``
    after every successful training step. We keep EMA off by default
    (``ema_decay = 0.0``) so tests can ignore the effect; real training uses
    e.g. ``0.9999``.
    """

    params: Any
    opt_state: Any
    ema_params: Any
    step: jnp.ndarray


@dataclass
class TrainMetrics:
    """What ``train`` returns: per-step loss trace and final state."""

    state: TrainState
    losses: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)


def init_train_state(
    model,
    sample_x_t: np.ndarray,
    sample_t: np.ndarray,
    sample_mask: np.ndarray,
    optimizer: optax.GradientTransformation,
    *,
    key: jax.Array,
) -> TrainState:
    """Initialise ``params``, ``opt_state``, and ``ema_params`` by a dummy call."""
    params = model.init(
        key,
        jnp.asarray(sample_x_t),
        jnp.asarray(sample_t),
        jnp.asarray(sample_mask),
    )
    opt_state = optimizer.init(params)
    ema_params = jax.tree_util.tree_map(jnp.copy, params)
    return TrainState(
        params=params,
        opt_state=opt_state,
        ema_params=ema_params,
        step=jnp.asarray(0, dtype=jnp.int32),
    )


def _make_loss_reduction(
    loss_fn: Callable[..., jnp.ndarray],
) -> Callable[[Any, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Reduce per-token loss to a scalar by summing and dividing by mask count."""

    def scalar_loss(params, model_apply, x_t, t, x1, mask):
        logits = model_apply(params, x_t, t, mask)
        per_token = loss_fn(logits, x1, x_t, t, mask)
        denom = jnp.maximum(mask.sum().astype(per_token.dtype), 1.0)
        return per_token.sum() / denom

    return scalar_loss


def make_train_step(
    model,
    process: SequenceProcess,
    optimizer: optax.GradientTransformation,
    *,
    ema_decay: float = 0.0,
) -> Callable[
    [TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[TrainState, jnp.ndarray],
]:
    """Build and jit a single training step for ``(model, process, optimizer)``."""
    loss_fn = process.jax_loss_fn()
    scalar_loss = _make_loss_reduction(loss_fn)
    apply_fn = model.apply

    @jax.jit
    def train_step(
        state: TrainState,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        x1: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> tuple[TrainState, jnp.ndarray]:
        def loss_wrt_params(p):
            return scalar_loss(p, apply_fn, x_t, t, x1, mask)

        loss, grads = jax.value_and_grad(loss_wrt_params)(state.params)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_ema = jax.tree_util.tree_map(
            lambda e, p: ema_decay * e + (1.0 - ema_decay) * p,
            state.ema_params,
            new_params,
        )
        new_state = TrainState(
            params=new_params,
            opt_state=new_opt_state,
            ema_params=new_ema,
            step=state.step + jnp.asarray(1, dtype=state.step.dtype),
        )
        return new_state, loss

    return train_step


def make_eval_step(
    model, process: SequenceProcess
) -> Callable[[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Build and jit a single evaluation step: returns mean per-token loss."""
    loss_fn = process.jax_loss_fn()
    scalar_loss = _make_loss_reduction(loss_fn)
    apply_fn = model.apply

    @jax.jit
    def eval_step(
        params: Any,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        x1: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        return scalar_loss(params, apply_fn, x_t, t, x1, mask)

    return eval_step


BatchIter = Iterable[tuple[np.ndarray, np.ndarray]]


def _corrupt_and_package(
    rng: np.random.Generator,
    process: SequenceProcess,
    ids: np.ndarray,
    mask: np.ndarray,
    *,
    time_scheme: str,
    t_max: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    t = sample_time(rng, ids.shape[0], scheme=time_scheme, t_max=t_max)
    state = process.corrupt(rng, ids, t, mask=mask)
    return (
        jnp.asarray(state.tensor),
        jnp.asarray(state.t),
        jnp.asarray(ids),
        jnp.asarray(mask),
    )


def train(
    model,
    process: SequenceProcess,
    optimizer: optax.GradientTransformation,
    batches: BatchIter,
    *,
    rng: np.random.Generator,
    key: jax.Array,
    ema_decay: float = 0.0,
    log_every: int = 50,
    eval_every: int = 0,
    eval_batches: BatchIter | None = None,
    time_scheme: str = "uniform",
    t_max: float = 1.0,
    callback: Callable[[int, float], None] | None = None,
) -> TrainMetrics:
    """Run a full training loop over ``batches`` once.

    Use ``itertools.cycle`` on your dataset iterator if you want multiple
    epochs; this function deliberately consumes the iterable as-is and does
    not own dataset lifecycle.
    """
    batch_iter = iter(batches)
    try:
        first_ids, first_mask = next(batch_iter)
    except StopIteration:
        raise ValueError("batches iterable produced no batches; cannot train.") from None

    sample_x_t, sample_t, sample_x1, sample_mask = _corrupt_and_package(
        rng, process, first_ids, first_mask, time_scheme=time_scheme, t_max=t_max
    )
    state = init_train_state(
        model, sample_x_t, sample_t, sample_mask, optimizer, key=key
    )

    train_step = make_train_step(model, process, optimizer, ema_decay=ema_decay)
    eval_step = make_eval_step(model, process) if eval_batches is not None else None

    metrics = TrainMetrics(state=state)

    def step_on(ids: np.ndarray, mask: np.ndarray, st: TrainState) -> tuple[TrainState, float]:
        x_t, t, x1, m = _corrupt_and_package(
            rng, process, ids, mask, time_scheme=time_scheme, t_max=t_max
        )
        new_st, loss = train_step(st, x_t, t, x1, m)
        return new_st, float(loss)

    state, first_loss = step_on(first_ids, first_mask, state)
    metrics.losses.append(first_loss)
    if callback:
        callback(0, first_loss)

    for i, (ids, mask) in enumerate(batch_iter, start=1):
        state, loss = step_on(ids, mask, state)
        metrics.losses.append(loss)
        if callback and (i % log_every == 0):
            callback(i, loss)
        if eval_step is not None and eval_every > 0 and i % eval_every == 0:
            eval_loss = evaluate(
                eval_step,
                state.params,
                process,
                eval_batches,  # type: ignore[arg-type]
                rng=rng,
                time_scheme=time_scheme,
                t_max=t_max,
            )
            metrics.eval_losses.append(eval_loss)

    metrics.state = state
    return metrics


def evaluate(
    eval_step: Callable[..., jnp.ndarray],
    params: Any,
    process: SequenceProcess,
    batches: BatchIter,
    *,
    rng: np.random.Generator,
    time_scheme: str = "uniform",
    t_max: float = 1.0,
) -> float:
    """Mean per-token loss across ``batches``, weighted by real token count."""
    total_loss = 0.0
    total_tokens = 0
    for ids, mask in batches:
        x_t, t, x1, m = _corrupt_and_package(
            rng, process, ids, mask, time_scheme=time_scheme, t_max=t_max
        )
        loss = float(eval_step(params, x_t, t, x1, m))
        n = int(mask.sum())
        total_loss += loss * n
        total_tokens += n
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


def cycle(factory: Callable[[], Iterable[Any]]) -> Iterator[Any]:
    """Infinitely cycle by re-invoking ``factory()`` each epoch.

    Unlike ``itertools.cycle`` this does not materialise items in memory.
    Unlike doing ``yield from iterable`` on a single generator, it correctly
    restarts from the beginning every pass by calling ``factory`` again to
    produce a fresh iterator.
    """
    while True:
        produced = False
        for item in factory():
            produced = True
            yield item
        if not produced:
            raise ValueError("factory() produced no items; would loop forever.")
