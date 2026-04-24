"""Numerical invariants of the mask-interpolant Discrete Flow Matching."""

from __future__ import annotations

import numpy as np
import pytest

from flowcompare.processes.dfm import DFMProcess, dfm_mask_loss

VOCAB = 12
MASK_ID = 3
BATCH = 4
LENGTH = 10


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture
def dfm() -> DFMProcess:
    return DFMProcess(vocab_size=VOCAB, mask_id=MASK_ID)


@pytest.fixture
def x1(rng: np.random.Generator) -> np.ndarray:
    # Sample from non-mask tokens so "not masked" is unambiguous.
    non_mask = np.array([v for v in range(VOCAB) if v != MASK_ID], dtype=np.int32)
    idx = rng.integers(0, len(non_mask), size=(BATCH, LENGTH))
    return non_mask[idx].astype(np.int32)


def test_rejects_mask_id_out_of_vocab() -> None:
    with pytest.raises(ValueError, match="mask_id"):
        DFMProcess(vocab_size=5, mask_id=10)


def test_prior_is_all_mask(dfm: DFMProcess, rng: np.random.Generator) -> None:
    state = dfm.sample_prior(rng, (BATCH, LENGTH))
    assert state.extras is not None
    tokens = state.extras["tokens"]
    assert (tokens == MASK_ID).all()
    # One-hot tensor mirrors tokens.
    assert np.allclose(state.tensor.argmax(axis=-1), MASK_ID)
    assert np.all(state.t == 0.0)


def test_corrupt_at_t0_is_all_mask(
    dfm: DFMProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    state = dfm.corrupt(rng, x1, np.float32(0.0))
    assert (state.extras["tokens"] == MASK_ID).all()


def test_corrupt_at_t1_is_data(
    dfm: DFMProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    state = dfm.corrupt(rng, x1, np.float32(1.0))
    assert (state.extras["tokens"] == x1).all()


def test_corrupt_keeps_fraction_t_on_average(
    dfm: DFMProcess, rng: np.random.Generator
) -> None:
    """Mask interpolant: P(z_t_i = x_1_i) = t (when x_1_i != MASK)."""
    K = VOCAB
    non_mask = np.array([v for v in range(K) if v != MASK_ID], dtype=np.int32)
    many = 2048
    idx = rng.integers(0, len(non_mask), size=(many, LENGTH))
    x = non_mask[idx].astype(np.int32)
    for t_val in (0.2, 0.5, 0.8):
        state = dfm.corrupt(rng, x, np.float32(t_val))
        frac = (state.extras["tokens"] == x).mean()
        assert abs(float(frac) - t_val) < 0.02


def test_corrupt_never_produces_other_tokens(
    dfm: DFMProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    """At corruption positions the result is exactly MASK, not another random token."""
    state = dfm.corrupt(rng, x1, np.float32(0.5))
    z = state.extras["tokens"]
    non_match = z != x1
    assert (z[non_match] == MASK_ID).all()


def test_loss_zero_when_no_masked_positions(
    dfm: DFMProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    """If the state has no MASK tokens, there is nothing to predict and loss is 0."""
    state = dfm.corrupt(rng, x1, np.float32(1.0))  # fully data
    logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
    loss = dfm.loss(logits, x1, state)
    assert np.all(loss == 0.0)


def test_loss_zero_for_perfect_prediction_at_masked_positions(
    dfm: DFMProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    """Infinite logit mass on x_1 gives cross-entropy = 0 at every position."""
    state = dfm.corrupt(rng, x1, np.float32(0.5))
    huge = 50.0
    logits = np.full((BATCH, LENGTH, VOCAB), -huge, dtype=np.float32)
    np.put_along_axis(logits, x1[..., None].astype(np.int64), huge, axis=-1)
    loss = dfm.loss(logits, x1, state)
    assert np.allclose(loss, 0.0, atol=1e-5)


def test_loss_positive_at_masked_positions_for_uniform_logits(
    dfm: DFMProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    state = dfm.corrupt(rng, x1, np.float32(0.5))
    logits = np.zeros((BATCH, LENGTH, VOCAB), dtype=np.float32)
    loss = dfm.loss(logits, x1, state)
    z = state.extras["tokens"]
    assert np.all(loss[z == MASK_ID] > 0.0)
    assert np.all(loss[z != MASK_ID] == 0.0)


def test_loss_weight_is_one_over_one_minus_t(
    dfm: DFMProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    """With the same z_t, loss at t=0.1 should be ~9x larger than at t=0.9 (ratio 1/0.9 : 1/0.1)."""
    # Force all positions masked so the comparison is clean.
    forced_state_tokens = np.full_like(x1, MASK_ID)
    logits = np.zeros((BATCH, LENGTH, VOCAB), dtype=np.float32)

    state_low = dfm.corrupt(rng, x1, np.float32(0.1))
    state_low = state_low.replace(
        tensor=state_low.tensor,
        t=np.full((BATCH,), 0.1, np.float32),
        extras={"tokens": forced_state_tokens.astype(np.int32)},
    )
    state_high = state_low.replace(t=np.full((BATCH,), 0.9, np.float32))

    l_low = dfm.loss(logits, x1, state_low).sum()
    l_high = dfm.loss(logits, x1, state_high).sum()
    ratio = l_low / l_high
    # l_low weight = 1/(1-0.1) = 1/0.9; l_high weight = 1/(1-0.9) = 1/0.1.
    # Expected l_high/l_low = (1/0.1) / (1/0.9) = 9.0; so l_low/l_high = 1/9.
    assert np.isclose(float(ratio), 1.0 / 9.0, rtol=1e-4)


def test_stateless_helper_matches_class(
    dfm: DFMProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    state = dfm.corrupt(rng, x1, np.float32(0.4))
    logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
    class_loss = dfm.loss(logits, x1, state)
    fn_loss = dfm_mask_loss(
        logits,
        x1,
        state.extras["tokens"],
        state.t,
        mask_id=MASK_ID,
        vocab_size=VOCAB,
    )
    assert np.allclose(class_loss, fn_loss, atol=1e-6)


def test_step_only_unmasks_never_remasks(
    dfm: DFMProcess, rng: np.random.Generator
) -> None:
    state = dfm.sample_prior(rng, (BATCH, LENGTH))
    logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
    prev_tokens = state.extras["tokens"].copy()
    state = dfm.step(rng, logits, state, 0.1)
    new_tokens = state.extras["tokens"]
    # A previously-unmasked position should never become masked again.
    was_data = prev_tokens != MASK_ID
    assert (new_tokens[was_data] == prev_tokens[was_data]).all()


def test_sampler_completes_with_oracle_network(
    dfm: DFMProcess, rng: np.random.Generator
) -> None:
    """Oracle logits predicting x_1 with high confidence yield full reconstruction."""
    non_mask = np.array([v for v in range(VOCAB) if v != MASK_ID], dtype=np.int32)
    idx = rng.integers(0, len(non_mask), size=(BATCH, LENGTH))
    x = non_mask[idx].astype(np.int32)
    state = dfm.sample_prior(rng, (BATCH, LENGTH))
    n_steps = 25
    dt = 1.0 / n_steps
    huge = 30.0
    for _ in range(n_steps):
        oracle_logits = np.full((BATCH, LENGTH, VOCAB), -huge, dtype=np.float32)
        np.put_along_axis(oracle_logits, x[..., None].astype(np.int64), huge, axis=-1)
        state = dfm.step(rng, oracle_logits, state, dt)
    assert np.all(state.t <= 1.0 + 1e-6)
    tokens = state.extras["tokens"]
    assert (tokens == x).all()


def test_sampler_final_state_has_no_masks_with_enough_steps(
    dfm: DFMProcess, rng: np.random.Generator
) -> None:
    """With the t=1 edge handling, the final state should contain no MASK tokens."""
    state = dfm.sample_prior(rng, (BATCH, LENGTH))
    n_steps = 50
    dt = 1.0 / n_steps
    for _ in range(n_steps):
        logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
        state = dfm.step(rng, logits, state, dt)
    assert (state.extras["tokens"] != MASK_ID).all()


def test_step_respects_state_contract() -> None:
    """Calling step on a hand-built state without extras should fail cleanly."""
    from flowcompare.processes.base import ProcessState

    dfm = DFMProcess(vocab_size=VOCAB, mask_id=MASK_ID)
    bad = ProcessState(
        tensor=np.zeros((1, 2, VOCAB), dtype=np.float32),
        t=np.zeros((1,), dtype=np.float32),
    )
    logits = np.zeros((1, 2, VOCAB), dtype=np.float32)
    with pytest.raises(ValueError, match="extras"):
        dfm.step(np.random.default_rng(0), logits, bad, 0.1)
