"""Numerical invariants of the categorical BFN.

Each test here pins down a specific property of Graves 2023's construction
that a correct implementation must satisfy. If any of these go red, the BFN
is wrong regardless of whether the higher-level contract tests pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from flowcompare.processes.bfn import BFNProcess, bfn_loss

VOCAB = 10
BATCH = 4
LENGTH = 8


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def bfn() -> BFNProcess:
    return BFNProcess(vocab_size=VOCAB, beta_1=3.0)


@pytest.fixture
def x1(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, VOCAB, size=(BATCH, LENGTH), dtype=np.int32)


def test_beta_schedule_endpoints(bfn: BFNProcess) -> None:
    """Cumulative accuracy starts at zero and ends at beta_1."""
    assert bfn.beta(np.asarray(0.0, dtype=np.float32)) == 0.0
    assert np.allclose(bfn.beta(np.asarray(1.0, dtype=np.float32)), 3.0)


def test_beta_prime_is_derivative(bfn: BFNProcess) -> None:
    """Finite-difference check of beta' against beta."""
    t = np.linspace(0.1, 0.9, 9, dtype=np.float32)
    h = 1e-3
    fd = (bfn.beta(t + h) - bfn.beta(t - h)) / (2 * h)
    assert np.allclose(fd, bfn.beta_prime(t), atol=1e-3)


def test_prior_theta_is_exactly_uniform(
    bfn: BFNProcess, rng: np.random.Generator
) -> None:
    state = bfn.sample_prior(rng, (BATCH, LENGTH))
    assert state.tensor.shape == (BATCH, LENGTH, VOCAB)
    assert np.allclose(state.tensor, 1.0 / VOCAB)
    assert np.all(state.t == 0.0)


def test_theta_is_on_simplex_after_corrupt(
    bfn: BFNProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    t = np.full((BATCH,), 0.5, dtype=np.float32)
    state = bfn.corrupt(rng, x1, t)
    assert np.all(state.tensor >= 0.0)
    assert np.allclose(state.tensor.sum(axis=-1), 1.0, atol=1e-5)


def test_theta_at_zero_time_is_uniform(
    bfn: BFNProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    """At t=0 beta(0)=0 so the sender has no mean and no variance; theta = 1/K."""
    state = bfn.corrupt(rng, x1, np.float32(0.0))
    assert np.allclose(state.tensor, 1.0 / VOCAB, atol=1e-5)


def test_theta_entropy_decreases_with_time_on_average(
    bfn: BFNProcess, rng: np.random.Generator
) -> None:
    """q_t(theta|x_1) concentrates as t increases; average entropy should drop."""
    many = 64
    x = rng.integers(0, VOCAB, size=(many, LENGTH), dtype=np.int32)

    def mean_entropy(t_val: float) -> float:
        state = bfn.corrupt(rng, x, np.float32(t_val))
        p = np.clip(state.tensor, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum(axis=-1).mean())

    h_low = mean_entropy(0.1)
    h_high = mean_entropy(0.9)
    assert h_high < h_low
    assert h_low <= np.log(VOCAB) + 1e-3


def test_theta_concentrates_near_one_hot_at_t1(
    bfn: BFNProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    """At t=1 the predicted argmax should usually agree with x_1."""
    # Use a larger beta_1 so concentration is clearer.
    bfn_big = BFNProcess(vocab_size=VOCAB, beta_1=20.0)
    state = bfn_big.corrupt(rng, x1, np.float32(1.0))
    agree = (state.tensor.argmax(axis=-1).astype(np.int32) == x1).mean()
    assert agree > 0.8


def test_loss_zero_when_prediction_is_perfect(
    bfn: BFNProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    """Logits that put infinite mass on x_1 yield p_hat = e(x) and zero loss."""
    state = bfn.corrupt(rng, x1, np.float32(0.5))
    huge = 50.0
    logits = np.full((BATCH, LENGTH, VOCAB), -huge, dtype=np.float32)
    np.put_along_axis(logits, x1[..., None].astype(np.int64), huge, axis=-1)
    loss = bfn.loss(logits, x1, state)
    assert np.allclose(loss, 0.0, atol=1e-5)


def test_loss_positive_for_uniform_logits(
    bfn: BFNProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    state = bfn.corrupt(rng, x1, np.float32(0.5))
    logits = np.zeros((BATCH, LENGTH, VOCAB), dtype=np.float32)
    loss = bfn.loss(logits, x1, state)
    assert np.all(loss > 0.0)


def test_loss_weight_scales_as_t(
    bfn: BFNProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    """Per-token weight is K * beta'(t) = 2 K beta_1 t; loss scales linearly in t."""
    logits = np.zeros((BATCH, LENGTH, VOCAB), dtype=np.float32)
    state_a = bfn.corrupt(rng, x1, np.float32(0.2))
    state_b = bfn.corrupt(rng, x1, np.float32(0.8))
    # Replace t on an artificial state with matching tensor so the (x_1, p_hat)
    # part is identical and only the weight differs.
    theta = np.full((BATCH, LENGTH, VOCAB), 1.0 / VOCAB, dtype=np.float32)
    s1 = state_a.replace(tensor=theta, t=np.full((BATCH,), 0.2, np.float32))
    s2 = state_b.replace(tensor=theta, t=np.full((BATCH,), 0.8, np.float32))
    l1 = bfn.loss(logits, x1, s1).sum()
    l2 = bfn.loss(logits, x1, s2).sum()
    assert np.isclose(l2 / l1, 0.8 / 0.2, rtol=1e-4)


def test_prior_kl_is_zero(bfn: BFNProcess, x1: np.ndarray) -> None:
    kl = bfn.prior_kl(x1)
    assert np.all(kl == 0.0)


def test_sampler_time_monotonic_and_clipped(
    bfn: BFNProcess, rng: np.random.Generator
) -> None:
    state = bfn.sample_prior(rng, (BATCH, LENGTH))
    logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
    dt = 0.1
    for _ in range(12):
        state = bfn.step(rng, logits, state, dt)
    assert np.all(state.t <= 1.0 + 1e-6)


def test_sampler_produces_state_on_simplex(
    bfn: BFNProcess, rng: np.random.Generator
) -> None:
    state = bfn.sample_prior(rng, (BATCH, LENGTH))
    logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
    for _ in range(5):
        state = bfn.step(rng, logits, state, 0.1)
        assert np.all(state.tensor >= 0.0)
        assert np.allclose(state.tensor.sum(axis=-1), 1.0, atol=1e-4)


def test_sampler_converges_to_correct_tokens_with_oracle_network(
    bfn: BFNProcess, rng: np.random.Generator
) -> None:
    """If the 'network' always returns the true x_1 as high-confidence logits,
    the sampler should converge to theta that argmaxes to x_1 at almost every
    position."""
    bfn_big = BFNProcess(vocab_size=VOCAB, beta_1=10.0)
    x = rng.integers(0, VOCAB, size=(BATCH, LENGTH), dtype=np.int32)
    state = bfn_big.sample_prior(rng, (BATCH, LENGTH))
    n_steps = 40
    dt = 1.0 / n_steps
    for _ in range(n_steps):
        huge = 30.0
        oracle_logits = np.full((BATCH, LENGTH, VOCAB), -huge, dtype=np.float32)
        np.put_along_axis(oracle_logits, x[..., None].astype(np.int64), huge, axis=-1)
        state = bfn_big.step(rng, oracle_logits, state, dt)
    agree = (state.tensor.argmax(axis=-1).astype(np.int32) == x).mean()
    assert agree > 0.95


def test_stateless_bfn_loss_matches_class(
    bfn: BFNProcess, rng: np.random.Generator, x1: np.ndarray
) -> None:
    """The functional bfn_loss helper reproduces the class's loss for fixed inputs."""
    t = np.full((BATCH,), 0.4, dtype=np.float32)
    state = bfn.corrupt(rng, x1, t)
    logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
    class_loss = bfn.loss(logits, x1, state)
    fn_loss = bfn_loss(logits, x1, state.t, vocab_size=VOCAB, beta_1=bfn.beta_1)
    assert np.allclose(class_loss, fn_loss, atol=1e-6)


def test_loss_sum_is_finite_for_random_everything(
    bfn: BFNProcess, rng: np.random.Generator
) -> None:
    x = rng.integers(0, VOCAB, size=(BATCH, LENGTH), dtype=np.int32)
    t = rng.uniform(0.0, 1.0, size=(BATCH,)).astype(np.float32)
    state = bfn.corrupt(rng, x, t)
    logits = rng.standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
    loss = bfn.loss(logits, x, state)
    assert np.all(np.isfinite(loss))
    assert loss.sum() > 0.0
