"""Discrete Flow Matching with the mask-based interpolant.

Follows Gat et al. 2024 (*Discrete Flow Matching*) and Campbell et al. 2024
(*Generative Flows on Discrete State-Spaces*). The chosen interpolant is the
"absorbing"/mask interpolant, equivalent in continuous time to the
absorbing-state discrete diffusion of Austin et al. and to a continuous-time
MaskGIT: at time ``t`` every position is independently either the true data
token (with probability ``t``) or ``MASK`` (with probability ``1-t``).

Only the mask interpolant is implemented in this module. The uniform
interpolant (``z_t_i ~ t * delta(x_1_i) + (1-t) * Uniform(vocab)``) produces
identical ``x_1``-prediction training mathematics but requires a CTMC sampler
over a full ``K x K`` rate matrix; it is a natural extension but not essential
for the head-to-head comparison this repo targets and is deferred.

Representation choices:

- ``state.tensor``: one-hot encoded tokens of shape ``(B, L, K)``, matching
  the shape that the shared Transformer backbone expects regardless of
  process. The canonical discrete state is kept in ``state.extras["tokens"]``
  as an ``int32`` array of shape ``(B, L)``; the one-hot tensor is just a
  view for the backbone.
- ``state.t``: per-example times.

Training math (mask interpolant, continuous time):

    L_DFM(x_1) = E_{t ~ U[0,1]} E_{z_t ~ q_t} [
        (1/(1-t)) * sum_{i: z_t_i = MASK} CE(p_theta(. | z_t, t)_i, x_1_i)
    ]

Sampler: tau-leaping. At each step of size ``dt``, every currently-masked
position is unmasked independently with probability ``dt / (1-t)`` (clipped
to 1 near ``t=1``). A position chosen to unmask is filled by sampling from
``softmax(net_logits)_i`` at that position.
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
    cum = np.cumsum(probs, axis=-1)
    u = rng.random(size=(*probs.shape[:-1], 1)).astype(np.float32)
    return (u < cum).argmax(axis=-1).astype(np.int32)


class DFMProcess(SequenceProcess):
    """Discrete Flow Matching with the absorbing-mask interpolant.

    Parameters
    ----------
    vocab_size :
        Number of tokens including the mask token.
    mask_id :
        Integer id used as the absorbing state. Default 3 matches
        ``ProteinTokenizer.MASK_ID``.
    loss_eps :
        Floor on ``1 - t`` when computing the ``1/(1-t)`` weight, to keep
        gradients finite as ``t`` approaches 1 during training. The sampler
        uses its own edge handling (clipping ``dt`` to ``1 - t``) and does
        not read this value.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        mask_id: int = 3,
        loss_eps: float = 1e-3,
    ) -> None:
        super().__init__(vocab_size)
        if not (0 <= mask_id < vocab_size):
            raise ValueError(
                f"mask_id {mask_id} must be in [0, {vocab_size})."
            )
        if loss_eps <= 0:
            raise ValueError(f"loss_eps must be positive, got {loss_eps}.")
        self.mask_id = int(mask_id)
        self.loss_eps = float(loss_eps)

    def _state_from_tokens(
        self,
        tokens: np.ndarray,
        t: np.ndarray,
        mask: MaskArray | None,
    ) -> ProcessState:
        tensor = _one_hot(tokens, self.vocab_size)
        return ProcessState(
            tensor=tensor,
            t=t.astype(np.float32),
            mask=mask,
            extras={"tokens": tokens.astype(np.int32)},
        )

    def sample_prior(
        self,
        rng: RNG,
        batch_shape: tuple[int, ...],
        *,
        mask: MaskArray | None = None,
    ) -> ProcessState:
        del rng  # mask interpolant prior is deterministic: all-MASK
        if len(batch_shape) != 2:
            raise ValueError(f"batch_shape must be (batch, length), got {batch_shape}.")
        batch, _length = batch_shape
        tokens = np.full(batch_shape, self.mask_id, dtype=np.int32)
        t = np.zeros((batch,), dtype=np.float32)
        return self._state_from_tokens(tokens, t, mask)

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
        t_arr = np.broadcast_to(np.asarray(t, dtype=np.float32), (x1.shape[0],)).copy()
        keep_prob = t_arr[:, None]
        u = rng.random(x1.shape).astype(np.float32)
        keep = u < keep_prob
        tokens = np.where(keep, x1, np.full_like(x1, self.mask_id))
        return self._state_from_tokens(tokens.astype(np.int32), t_arr, mask)

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
        if state.extras is None or "tokens" not in state.extras:
            raise ValueError(
                "DFM state must carry extras['tokens']; construct via corrupt()/sample_prior()."
            )
        z = state.extras["tokens"]
        log_probs = _log_softmax(net_logits.astype(np.float32))
        picked = np.take_along_axis(
            log_probs, x1[..., None].astype(np.int64), axis=-1
        )[..., 0]
        nll = -picked  # (B, L)
        is_masked = (z == self.mask_id).astype(np.float32)
        weight = 1.0 / np.maximum(1.0 - state.t.astype(np.float32), self.loss_eps)  # (B,)
        per_token = nll * is_masked * weight[:, None]
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
        if state.extras is None or "tokens" not in state.extras:
            raise ValueError(
                "DFM state must carry extras['tokens']; construct via sample_prior()."
            )
        z = state.extras["tokens"]
        t = state.t.astype(np.float32)
        t_next = np.clip(t + np.float32(dt), 0.0, 1.0)
        step_dt = t_next - t  # (B,)

        # Per-example unmasking probability: step_dt / (1-t), clipped to [0, 1].
        # Using max(1-t, step_dt) in the denominator guarantees p <= 1 even at
        # the t=1 boundary (there p -> 1 exactly, unmasking any remaining
        # positions on the final step).
        denom = np.maximum(1.0 - t, step_dt)
        denom = np.where(denom > 0, denom, 1.0)
        p_unmask = np.clip(step_dt / denom, 0.0, 1.0)[:, None]  # (B, 1)

        is_masked = z == self.mask_id
        r = rng.random(z.shape).astype(np.float32)
        choose = is_masked & (r < p_unmask)

        # When unmasking, sample from the network's predicted distribution
        # over NON-mask tokens only: a well-trained network will not place
        # mass on MASK (real data never contains it), but a random or
        # under-trained network will, which otherwise stalls the sampler.
        # This is the standard inference-time trick for absorbing-state
        # discrete diffusion / DFM mask interpolants.
        logits = net_logits.astype(np.float32).copy()
        logits[..., self.mask_id] = -np.inf
        p_hat = _softmax(logits)
        new_tokens = _sample_categorical(rng, p_hat)
        z_next = np.where(choose, new_tokens, z).astype(np.int32)

        return self._state_from_tokens(z_next, t_next, state.mask)

    def prior_kl(
        self,
        x1: TokenArray,
        *,
        mask: MaskArray | None = None,
    ) -> LogitArray:
        # The continuous-time ELBO with kappa(t) = t absorbs the prior
        # contribution into the integrated loss: kappa'(t)/(1 - kappa(t)) = 1/(1-t)
        # integrates to log(1/(1-t)) which diverges at t=1 but is compensated
        # by the mask-indicator weighting. Following Gat et al., we return zero
        # here and rely on the training loop to sample t sufficiently far from
        # the t=1 boundary (or use loss_eps clipping).
        kl = np.zeros(x1.shape, dtype=np.float32)
        if mask is not None:
            kl = np.where(mask, kl, 0.0)
        return kl


    def jax_loss_fn(self):
        """Pure-JAX training-loss closure.

        Signature: ``fn(logits, x1, x_t, t, mask) -> per_token_loss``. Reads
        the current tokens ``z_t`` as ``argmax(x_t)`` so the function remains
        pure (no extras dict needed). This is exact because ``x_t`` is
        constructed as a one-hot of tokens by :meth:`corrupt` and
        :meth:`sample_prior`.
        """
        import jax.numpy as jnp
        from jax.nn import log_softmax

        mask_id = self.mask_id
        loss_eps = self.loss_eps

        def fn(logits, x1, x_t, t, mask):
            log_probs = log_softmax(logits, axis=-1)
            picked = jnp.take_along_axis(log_probs, x1[..., None], axis=-1)[..., 0]
            nll = -picked
            z_t = jnp.argmax(x_t, axis=-1)
            is_masked = (z_t == mask_id).astype(logits.dtype)
            weight = 1.0 / jnp.maximum(1.0 - t, loss_eps)
            per_token = nll * is_masked * weight[:, None]
            return jnp.where(mask, per_token, 0.0)

        return fn


def dfm_mask_loss(
    net_logits: np.ndarray,
    x1: np.ndarray,
    z_t: np.ndarray,
    t: np.ndarray,
    *,
    mask_id: int,
    vocab_size: int,
    mask: np.ndarray | None = None,
    loss_eps: float = 1e-3,
) -> np.ndarray:
    """Stateless DFM mask-interpolant loss for tests and parity checks.

    Returns per-token loss of shape ``(B, L)``; zero at positions that are
    already unmasked.
    """
    if net_logits.shape[-1] != vocab_size:
        raise ValueError(
            f"Vocab mismatch: logits {net_logits.shape[-1]} vs vocab_size {vocab_size}."
        )
    log_probs = _log_softmax(net_logits.astype(np.float32))
    picked = np.take_along_axis(log_probs, x1[..., None].astype(np.int64), axis=-1)[..., 0]
    nll = -picked
    is_masked = (z_t == mask_id).astype(np.float32)
    weight = 1.0 / np.maximum(1.0 - t.astype(np.float32), loss_eps)
    per_token = nll * is_masked * weight[:, None]
    if mask is not None:
        per_token = np.where(mask, per_token, 0.0)
    return per_token.astype(np.float32)
