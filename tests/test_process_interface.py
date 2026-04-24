"""Contract tests for the ``SequenceProcess`` interface.

Any concrete subclass must satisfy these. They do not check numerical
correctness of any specific method (BFN/DFM have their own math tests); they
verify that the unified interface holds: shapes, dtypes, mask propagation,
time-coordinate invariants, and that a single training+sampling round-trip
executes end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from flowcompare.processes import (
    BFNProcess,
    DFMProcess,
    IdentityProcess,
    ProcessState,
    SequenceProcess,
)

VOCAB = 25
BATCH = 3
LENGTH = 7


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _make_identity() -> SequenceProcess:
    return IdentityProcess(vocab_size=VOCAB)


def _make_bfn() -> SequenceProcess:
    return BFNProcess(vocab_size=VOCAB, beta_1=3.0)


def _make_dfm() -> SequenceProcess:
    return DFMProcess(vocab_size=VOCAB, mask_id=3)


@pytest.fixture(
    params=[_make_identity, _make_bfn, _make_dfm],
    ids=["IdentityProcess", "BFNProcess", "DFMProcess"],
)
def process(request) -> SequenceProcess:
    return request.param()


@pytest.fixture
def x1(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, VOCAB, size=(BATCH, LENGTH), dtype=np.int32)


@pytest.fixture
def pad_mask() -> np.ndarray:
    m = np.ones((BATCH, LENGTH), dtype=bool)
    m[:, -2:] = False
    return m


def test_vocab_size_is_positive(process: SequenceProcess) -> None:
    assert process.vocab_size > 0


def test_cannot_instantiate_abc() -> None:
    with pytest.raises(TypeError):
        SequenceProcess(vocab_size=VOCAB)  # type: ignore[abstract]


def test_sample_prior_returns_process_state(
    process: SequenceProcess, rng: np.random.Generator
) -> None:
    state = process.sample_prior(rng, (BATCH, LENGTH))
    assert isinstance(state, ProcessState)
    assert state.t.shape == (BATCH,)
    assert np.all(state.t == 0.0)
    assert state.tensor.shape[0] == BATCH
    assert state.tensor.shape[1] == LENGTH


def test_corrupt_broadcasts_scalar_time(
    process: SequenceProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    state = process.corrupt(rng, x1, np.float32(0.5))
    assert state.t.shape == (BATCH,)
    assert np.allclose(state.t, 0.5)


def test_corrupt_accepts_per_example_time(
    process: SequenceProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    t = np.linspace(0.1, 0.9, BATCH, dtype=np.float32)
    state = process.corrupt(rng, x1, t)
    assert np.allclose(state.t, t)


def test_corrupt_propagates_mask(
    process: SequenceProcess,
    rng: np.random.Generator,
    x1: np.ndarray,
    pad_mask: np.ndarray,
) -> None:
    state = process.corrupt(rng, x1, np.float32(0.5), mask=pad_mask)
    assert state.mask is not None
    assert state.mask.shape == pad_mask.shape
    assert (state.mask == pad_mask).all()


def test_to_network_input_has_shape_compatible_with_loss(
    process: SequenceProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    state = process.corrupt(rng, x1, np.float32(0.3))
    inp = process.to_network_input(state)
    assert inp.shape[:2] == (BATCH, LENGTH)


def test_loss_returns_per_token_array(
    process: SequenceProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    state = process.corrupt(rng, x1, np.float32(0.3))
    logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
    loss = process.loss(logits, x1, state)
    assert loss.shape == (BATCH, LENGTH)
    assert loss.dtype == np.float32
    assert np.all(loss >= 0.0)


def test_loss_respects_mask(
    process: SequenceProcess,
    rng: np.random.Generator,
    x1: np.ndarray,
    pad_mask: np.ndarray,
) -> None:
    state = process.corrupt(rng, x1, np.float32(0.3), mask=pad_mask)
    logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
    loss = process.loss(logits, x1, state, mask=pad_mask)
    assert (loss[~pad_mask] == 0.0).all()


def test_loss_rejects_vocab_mismatch(
    process: SequenceProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    state = process.corrupt(rng, x1, np.float32(0.3))
    bad_logits = rng.standard_normal((BATCH, LENGTH, VOCAB + 1)).astype(np.float32)
    with pytest.raises(ValueError, match="Vocab"):
        process.loss(bad_logits, x1, state)


def test_step_advances_time_monotonically(
    process: SequenceProcess, rng: np.random.Generator
) -> None:
    state = process.sample_prior(rng, (BATCH, LENGTH))
    logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
    dt = 0.1
    next_state = process.step(rng, logits, state, dt)
    assert np.all(next_state.t >= state.t)
    assert np.all(next_state.t <= 1.0 + 1e-6)


def test_full_sampling_loop_runs(
    process: SequenceProcess, rng: np.random.Generator
) -> None:
    """Integrate from t=0 to t=1 with random logits at every step."""
    state = process.sample_prior(rng, (BATCH, LENGTH))
    n_steps = 20
    dt = 1.0 / n_steps
    for _ in range(n_steps):
        logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
        state = process.step(rng, logits, state, dt)
    assert np.all(state.t <= 1.0 + 1e-6)
    assert state.tensor.shape[0] == BATCH


def test_prior_kl_shape_and_mask(
    process: SequenceProcess, x1: np.ndarray, pad_mask: np.ndarray
) -> None:
    kl = process.prior_kl(x1, mask=pad_mask)
    assert kl.shape == x1.shape
    assert (kl[~pad_mask] == 0.0).all()
    # Prior KL is non-negative; some processes (e.g. BFN with data-independent
    # uniform prior at t=0) legitimately have zero KL at every real position.
    assert np.all(kl[pad_mask] >= 0.0)


def test_process_state_replace_creates_new_instance() -> None:
    t = np.zeros((2,), dtype=np.float32)
    tensor = np.zeros((2, 3, 4), dtype=np.float32)
    state = ProcessState(tensor=tensor, t=t)
    new = state.replace(t=np.ones((2,), dtype=np.float32))
    assert new is not state
    assert np.all(state.t == 0.0)
    assert np.all(new.t == 1.0)


def test_identity_process_is_noise_free(
    rng: np.random.Generator, x1: np.ndarray
) -> None:
    proc = IdentityProcess(vocab_size=VOCAB)
    state = proc.corrupt(rng, x1, np.float32(0.5))
    recovered = state.tensor.argmax(axis=-1).astype(np.int32)
    assert (recovered == x1).all()
