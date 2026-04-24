"""Categorical Bayesian Flow Network.

Follows Graves et al. 2023, *Bayesian Flow Networks*, section 6 (discrete /
categorical data). The continuous-time loss and the ``n``-step sampler here
are the direct analogues of Algorithms 9 and 10 of the paper, specialised to
the quadratic accuracy schedule ``beta(t) = beta_1 * t**2`` that is the
standard choice in the protein BFN variants (InstaDeep's
``protein-sequence-bfn`` uses the same form).

Key objects:

- ``state.tensor``: ``theta``, the input distribution parameters; a point on
  the ``K``-simplex at every sequence position. Shape ``(B, L, K)``.
- ``state.t``: per-example time in ``[0, 1]``. At ``t = 0`` the prior is
  uniform ``theta_0 = 1/K``; at ``t = 1`` ``theta`` concentrates near the
  one-hot data.
- ``beta(t)``: cumulative accuracy; ``beta(0) = 0``, ``beta(1) = beta_1``.
- ``beta_prime(t) = 2 * beta_1 * t``: instantaneous accuracy flux; appears as
  the loss weight in the continuous-time bound.

Relationship to the unified ``SequenceProcess`` interface: ``corrupt`` draws a
one-shot sample from ``q_t(theta | x_1)`` (sender + Bayesian update starting
from the uniform prior); ``step`` performs one Bayesian update of ``theta``
given the network's current ``x_1`` prediction (Algorithm 10, inner loop);
``loss`` is the continuous-time bound's integrand at a sampled ``(t, theta_t)``.
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

_LOG_EPS = 1e-30


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _one_hot(tokens: np.ndarray, K: int) -> np.ndarray:
    out = np.zeros((*tokens.shape, K), dtype=np.float32)
    np.put_along_axis(out, tokens[..., None].astype(np.int64), 1.0, axis=-1)
    return out


def _sample_categorical(rng: RNG, probs: np.ndarray) -> np.ndarray:
    """Per-position categorical sample from ``probs``; shape ``(..., K)`` -> ``(...)``."""
    cum = np.cumsum(probs, axis=-1)
    u = rng.random(size=(*probs.shape[:-1], 1)).astype(np.float32)
    idx = (u < cum).argmax(axis=-1)
    return idx.astype(np.int32)


class BFNProcess(SequenceProcess):
    """Categorical Bayesian Flow Network with quadratic accuracy schedule.

    Parameters
    ----------
    vocab_size :
        Categorical cardinality ``K``.
    beta_1 :
        Terminal accuracy ``beta(1)``. Larger values drive ``theta_1`` closer
        to the true one-hot; InstaDeep's ProtBFN uses values in the single
        digits. Loss weight is proportional to ``beta_1`` as well.
    """

    def __init__(self, vocab_size: int, *, beta_1: float = 3.0) -> None:
        super().__init__(vocab_size)
        if beta_1 <= 0:
            raise ValueError(f"beta_1 must be positive, got {beta_1}.")
        self.beta_1 = float(beta_1)

    def beta(self, t: np.ndarray) -> np.ndarray:
        return self.beta_1 * (t.astype(np.float32) ** 2)

    def beta_prime(self, t: np.ndarray) -> np.ndarray:
        return 2.0 * self.beta_1 * t.astype(np.float32)

    def sample_prior(
        self,
        rng: RNG,
        batch_shape: tuple[int, ...],
        *,
        mask: MaskArray | None = None,
    ) -> ProcessState:
        del rng  # uniform prior is deterministic
        if len(batch_shape) != 2:
            raise ValueError(f"batch_shape must be (batch, length), got {batch_shape}.")
        batch, length = batch_shape
        K = self.vocab_size
        theta = np.full((batch, length, K), 1.0 / K, dtype=np.float32)
        t = np.zeros((batch,), dtype=np.float32)
        return ProcessState(tensor=theta, t=t, mask=mask)

    def corrupt(
        self,
        rng: RNG,
        x1: TokenArray,
        t: TimeArray,
        *,
        mask: MaskArray | None = None,
    ) -> ProcessState:
        if x1.ndim != 2:
            raise ValueError(f"x1 must be (batch, length), got shape {x1.shape}.")
        K = self.vocab_size
        t_arr = np.broadcast_to(np.asarray(t, dtype=np.float32), (x1.shape[0],)).copy()
        beta_t = self.beta(t_arr)  # (B,)
        e_x = _one_hot(x1, K)  # (B, L, K)
        beta_bl = beta_t[:, None, None]
        mean = beta_bl * (K * e_x - 1.0)
        std = np.sqrt(np.maximum(beta_bl * K, 0.0))
        noise = rng.standard_normal(size=e_x.shape).astype(np.float32)
        y = mean + std * noise
        theta = _softmax(y)
        return ProcessState(tensor=theta, t=t_arr, mask=mask)

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
        if net_logits.shape[:-1] != x1.shape:
            raise ValueError(
                f"Shape mismatch: logits {net_logits.shape} vs tokens {x1.shape}."
            )
        if net_logits.shape[-1] != self.vocab_size:
            raise ValueError(
                f"Vocab mismatch: logits {net_logits.shape[-1]} vs process {self.vocab_size}."
            )
        K = self.vocab_size
        p_hat = _softmax(net_logits.astype(np.float32))
        e_x = _one_hot(x1, K)
        sq = ((e_x - p_hat) ** 2).sum(axis=-1)  # (B, L)
        # Continuous-time loss integrand (Graves 2023, Thm 4.2): the per-token
        # weight at time t is K * beta'(t) = 2 K beta_1 t. t is carried on the
        # state from corrupt(); the training loop samples t ~ U[0, 1] and
        # calls corrupt + loss once per example, giving a Monte Carlo
        # estimator of L^infty.
        w = K * self.beta_prime(state.t.astype(np.float32))  # (B,)
        per_token = sq * w[:, None]
        if mask is not None:
            per_token = np.where(mask, per_token, 0.0)
        return per_token.astype(np.float32)

    def step(
        self,
        rng: RNG,
        net_logits: LogitArray,
        state: ProcessState,
        dt: float,
    ) -> ProcessState:
        K = self.vocab_size
        theta = state.tensor
        t = state.t.astype(np.float32)
        t_next = np.clip(t + np.float32(dt), 0.0, 1.0)
        d_beta = self.beta(t_next) - self.beta(t)  # (B,)
        p_hat = _softmax(net_logits.astype(np.float32))
        x_hat = _sample_categorical(rng, p_hat)  # (B, L)
        e_hat = _one_hot(x_hat, K)
        d_bl = d_beta[:, None, None]
        mean = d_bl * (K * e_hat - 1.0)
        std = np.sqrt(np.maximum(d_bl * K, 0.0))
        noise = rng.standard_normal(size=e_hat.shape).astype(np.float32)
        y = mean + std * noise
        theta_next = _softmax(np.log(np.maximum(theta, _LOG_EPS)) + y)
        return state.replace(tensor=theta_next.astype(np.float32), t=t_next)

    def prior_kl(
        self,
        x1: TokenArray,
        *,
        mask: MaskArray | None = None,
    ) -> LogitArray:
        # For BFN with beta(0) = 0, the prior theta_0 is exactly uniform,
        # independent of x_1. The continuous-time ELBO therefore has no
        # explicit prior KL term: it is absorbed into the integrated loss.
        kl = np.zeros(x1.shape, dtype=np.float32)
        if mask is not None:
            kl = np.where(mask, kl, 0.0)
        return kl

    def jax_loss_fn(self):
        """Pure-JAX training-loss closure.

        Signature: ``fn(logits, x1, x_t, t, mask) -> per_token_loss``. The
        ``x_t`` input is the simplex parameter ``theta``; the BFN loss does
        not consume it directly because the continuous-time estimator is
        defined entirely in terms of ``(t, x_1, theta_network_output)``.
        """
        import jax.numpy as jnp
        from jax.nn import one_hot, softmax

        K = self.vocab_size
        beta_1 = self.beta_1

        def fn(logits, x1, x_t, t, mask):
            del x_t
            p_hat = softmax(logits, axis=-1)
            e_x = one_hot(x1, K, dtype=logits.dtype)
            sq = jnp.sum((e_x - p_hat) ** 2, axis=-1)
            w = K * 2.0 * beta_1 * t
            per_token = sq * w[:, None]
            return jnp.where(mask, per_token, 0.0)

        return fn


# ---------------------------------------------------------------------------
# Public top-level helpers reusing the math in a functional style. Useful for
# parity tests against reference implementations and for the theory writeup's
# numerical checks.
# ---------------------------------------------------------------------------


def bfn_loss(
    net_logits: np.ndarray,
    x1: np.ndarray,
    t: np.ndarray,
    *,
    vocab_size: int,
    beta_1: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Stateless BFN loss as a single function for tests and parity checks.

    Returns per-token loss of shape ``(B, L)``.
    """
    K = vocab_size
    p_hat = _softmax(net_logits.astype(np.float32))
    e_x = _one_hot(x1, K)
    sq = ((e_x - p_hat) ** 2).sum(axis=-1)
    w = K * 2.0 * beta_1 * t.astype(np.float32)
    per_token = sq * w[:, None]
    if mask is not None:
        per_token = np.where(mask, per_token, 0.0)
    return per_token.astype(np.float32)
