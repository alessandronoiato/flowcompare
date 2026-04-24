"""Trivial ``SequenceProcess`` used for tests and scaffolding.

``IdentityProcess`` satisfies the interface without introducing any actual
noise schedule or sampler dynamics: the prior is uniform over tokens, the
corruption is the identity (``z_t = x_1`` for all ``t``), and the loss is
plain cross-entropy against ``x_1``. It is not a useful generative model; its
job is to let the shared training and sampling infrastructure be exercised
before the real BFN and DFM processes land.
"""

from __future__ import annotations

import numpy as np

from flowcompare.processes.base import (
    RNG,
    LogitArray,
    MaskArray,
    ProcessState,
    SequenceProcess,
    StateArray,
    TimeArray,
    TokenArray,
)


def _one_hot(tokens: TokenArray, vocab_size: int) -> StateArray:
    out = np.zeros((*tokens.shape, vocab_size), dtype=np.float32)
    np.put_along_axis(out, tokens[..., None].astype(np.int64), 1.0, axis=-1)
    return out


def _log_softmax(logits: LogitArray) -> LogitArray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class IdentityProcess(SequenceProcess):
    """Noise-free stub. Intermediate state equals the data at every ``t``."""

    def sample_prior(
        self,
        rng: RNG,
        batch_shape: tuple[int, ...],
        *,
        mask: MaskArray | None = None,
    ) -> ProcessState:
        if len(batch_shape) != 2:
            raise ValueError(f"batch_shape must be (batch, length), got {batch_shape}.")
        tokens = rng.integers(0, self.vocab_size, size=batch_shape, dtype=np.int32)
        tensor = _one_hot(tokens, self.vocab_size)
        t = np.zeros((batch_shape[0],), dtype=np.float32)
        return ProcessState(tensor=tensor, t=t, mask=mask)

    def corrupt(
        self,
        rng: RNG,
        x1: TokenArray,
        t: TimeArray,
        *,
        mask: MaskArray | None = None,
    ) -> ProcessState:
        del rng  # corruption is deterministic here
        if x1.ndim != 2:
            raise ValueError(f"x1 must be (batch, length), got shape {x1.shape}.")
        t = np.broadcast_to(np.asarray(t, dtype=np.float32), (x1.shape[0],)).copy()
        tensor = _one_hot(x1, self.vocab_size)
        return ProcessState(tensor=tensor, t=t, mask=mask)

    def to_network_input(self, state: ProcessState) -> StateArray:
        return state.tensor

    def loss(
        self,
        net_logits: LogitArray,
        x1: TokenArray,
        state: ProcessState,
        *,
        mask: MaskArray | None = None,
    ) -> LogitArray:
        del state  # IdentityProcess loss is independent of the state
        if net_logits.shape[:-1] != x1.shape:
            raise ValueError(
                f"Shape mismatch: logits {net_logits.shape} vs tokens {x1.shape}."
            )
        if net_logits.shape[-1] != self.vocab_size:
            raise ValueError(
                f"Vocab mismatch: logits {net_logits.shape[-1]} vs process {self.vocab_size}."
            )
        log_probs = _log_softmax(net_logits)
        picked = np.take_along_axis(log_probs, x1[..., None].astype(np.int64), axis=-1)[..., 0]
        nll = -picked
        if mask is not None:
            nll = np.where(mask, nll, 0.0)
        return nll.astype(np.float32)

    def step(
        self,
        rng: RNG,
        net_logits: LogitArray,
        state: ProcessState,
        dt: float,
    ) -> ProcessState:
        del rng
        tokens = net_logits.argmax(axis=-1).astype(np.int32)
        tensor = _one_hot(tokens, self.vocab_size)
        t_next = np.clip(state.t + np.float32(dt), 0.0, 1.0)
        return state.replace(tensor=tensor, t=t_next)

    def prior_kl(
        self,
        x1: TokenArray,
        *,
        mask: MaskArray | None = None,
    ) -> LogitArray:
        kl = np.full(x1.shape, np.log(self.vocab_size), dtype=np.float32)
        if mask is not None:
            kl = np.where(mask, kl, 0.0)
        return kl

    def jax_loss_fn(self):
        import jax.numpy as jnp
        from jax.nn import log_softmax

        def fn(logits, x1, x_t, t, mask):
            del x_t, t
            log_probs = log_softmax(logits, axis=-1)
            picked = jnp.take_along_axis(log_probs, x1[..., None], axis=-1)[..., 0]
            nll = -picked
            return jnp.where(mask, nll, 0.0)

        return fn
