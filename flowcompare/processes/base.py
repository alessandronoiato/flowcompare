"""Abstract interface shared by all sequence generative processes.

The unifying view: every method implemented in this repo defines a
continuous-time process ``q_t(z_t | x_1)`` interpolating between a simple
prior at ``t=0`` and the data distribution at ``t=1``, together with a neural
network that predicts clean tokens ``x_1`` from a noised state ``z_t``.

- **Bayesian Flow Networks** choose ``z_t`` to be continuous parameters of a
  categorical distribution on the simplex; ``q_t`` adds Gaussian noise to
  those parameters with a time-varying accuracy schedule.
- **Discrete Flow Matching** chooses ``z_t`` to be discrete tokens and defines
  ``q_t`` through a continuous-time Markov chain (for example, the mask-based
  or uniform interpolant).

Every downstream difference (loss form, sampling algorithm, NFE economics,
compatibility with infilling) is a consequence of that single choice. This
module encodes the shared interface so those differences become isolated,
testable, and directly comparable.

Arrays here are typed as ``numpy.ndarray`` while the repo grows; concrete
subclasses will switch to ``jax.Array`` in a later slice without altering
these method signatures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

TokenArray = np.ndarray
LogitArray = np.ndarray
StateArray = np.ndarray
TimeArray = np.ndarray
MaskArray = np.ndarray
RNG = np.random.Generator


@dataclass(frozen=True)
class ProcessState:
    """Intermediate state ``z_t`` along the generative path.

    Attributes
    ----------
    tensor :
        The raw state representation. For BFN this is a simplex-valued tensor
        of shape ``(batch, length, vocab)``. For DFM this is an ``int32``
        token tensor of shape ``(batch, length)``. Subclasses document their
        own shape/dtype contract.
    t :
        Time coordinate(s). Shape ``(batch,)`` or scalar. ``t=0`` is the
        prior, ``t=1`` is data. Monotonically increasing during sampling.
    mask :
        Optional boolean mask of shape ``(batch, length)`` indicating real
        tokens (``True``) vs padding (``False``). ``None`` means "no padding,
        treat every position as real".
    extras :
        Process-specific side information (for example, BFN accuracies or DFM
        interpolant flags). Opaque to callers outside the owning process.
    """

    tensor: StateArray
    t: TimeArray
    mask: MaskArray | None = None
    extras: dict[str, Any] | None = None

    def replace(self, **changes: Any) -> ProcessState:
        """Return a new state with the given fields overridden."""
        base: dict[str, Any] = dict(
            tensor=self.tensor, t=self.t, mask=self.mask, extras=self.extras
        )
        base.update(changes)
        return ProcessState(**base)


class SequenceProcess(ABC):
    """Abstract continuous-time process over token sequences.

    A subclass fully specifies a generative method by implementing the six
    methods below. The shared training loop and the shared sampler in the
    ``training`` and ``sampling`` modules consume any conforming subclass
    without further modification.
    """

    def __init__(self, vocab_size: int) -> None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        self._vocab_size = int(vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @abstractmethod
    def sample_prior(
        self,
        rng: RNG,
        batch_shape: tuple[int, ...],
        *,
        mask: MaskArray | None = None,
    ) -> ProcessState:
        """Draw an initial state ``z_0 ~ q_0``.

        Parameters
        ----------
        rng :
            Source of randomness.
        batch_shape :
            Tuple ``(batch, length)`` specifying the state grid.
        mask :
            Optional validity mask propagated into the returned state.
        """

    @abstractmethod
    def corrupt(
        self,
        rng: RNG,
        x1: TokenArray,
        t: TimeArray,
        *,
        mask: MaskArray | None = None,
    ) -> ProcessState:
        """Sample ``z_t ~ q_t(.|x_1)`` for training.

        ``x1`` is an ``int32`` token array of shape ``(batch, length)``.
        ``t`` is a float array of per-example times in ``[0, 1]``.
        """

    @abstractmethod
    def to_network_input(self, state: ProcessState) -> StateArray:
        """Transform a state into the tensor the backbone consumes.

        The backbone is process-agnostic; it sees a single continuous tensor
        per state plus the time embedding. BFN passes simplex parameters
        directly; DFM one-hot encodes its token state.
        """

    @abstractmethod
    def loss(
        self,
        net_logits: LogitArray,
        x1: TokenArray,
        state: ProcessState,
        *,
        mask: MaskArray | None = None,
    ) -> LogitArray:
        """Per-token training loss given the network's ``x_1``-prediction logits.

        Returns an array of shape ``(batch, length)`` so callers can apply
        padding masks and reduce as needed. The continuous-time ELBOs of BFN
        and DFM both reduce to a per-token reweighted cross-entropy of this
        form, differing only in the time-dependent weighting.
        """

    @abstractmethod
    def step(
        self,
        rng: RNG,
        net_logits: LogitArray,
        state: ProcessState,
        dt: float,
    ) -> ProcessState:
        """Advance ``state`` by one sampler step of size ``dt``.

        Consumes the network's current ``x_1``-prediction logits and returns
        the next state. BFN performs a Bayesian parameter update; DFM applies
        a CTMC transition using the induced rate matrix.
        """

    @abstractmethod
    def prior_kl(self, x1: TokenArray, *, mask: MaskArray | None = None) -> LogitArray:
        """KL from ``q_1(.|x_1)`` to the prior ``q_0``, per token.

        Needed to complete the continuous-time ELBO: ``ELBO = -integral(loss dt)
        - prior_kl + reconstruction``. Returns shape ``(batch, length)``.
        """

    @abstractmethod
    def jax_loss_fn(self):
        """Return a pure-JAX training loss closure.

        The returned callable has signature
        ``fn(logits, x1, x_t, t, mask) -> per_token_loss`` with all arguments
        being ``jax.Array`` and the result shape ``(B, L)``. The training loop
        calls this inside ``jit`` / ``jax.grad``.

        The numpy-based :meth:`loss` is the ground-truth reference (tested for
        each process in its math-tests file); :meth:`jax_loss_fn` reproduces
        the same mathematics in a form that autodiffs and compiles.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vocab_size={self.vocab_size})"
