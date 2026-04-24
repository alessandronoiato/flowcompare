"""Tests for the evaluation suite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from flowcompare.data.collate import TokenizedDataset, iterate_batches
from flowcompare.data.synthetic import generate_synthetic_proteins
from flowcompare.eval import esm, esmfold
from flowcompare.eval.diversity import mean_pairwise_identity, pairwise_identity
from flowcompare.eval.novelty import max_identity_to_train
from flowcompare.eval.pareto import pareto_sweep
from flowcompare.eval.perplexity import compute_held_out_loss
from flowcompare.models.transformer import TimeConditionedTransformer
from flowcompare.processes.bfn import BFNProcess
from flowcompare.processes.dfm import DFMProcess
from flowcompare.tokenizer import ProteinTokenizer
from flowcompare.training.trainer import init_train_state

VOCAB = 25
BATCH = 4
LENGTH = 32


@pytest.fixture
def tokenizer() -> ProteinTokenizer:
    return ProteinTokenizer()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def test_pairwise_identity_identical_sequences() -> None:
    mat = pairwise_identity(["ACDE", "ACDE", "ACDE"])
    assert np.allclose(mat, 1.0)


def test_pairwise_identity_distinct_sequences() -> None:
    mat = pairwise_identity(["ACDE", "FGHI"])
    assert mat[0, 0] == 1.0
    assert mat[1, 1] == 1.0
    assert mat[0, 1] == 0.0
    assert mat[1, 0] == 0.0


def test_pairwise_identity_partial_match() -> None:
    mat = pairwise_identity(["ACDE", "ACXY"])
    assert np.isclose(mat[0, 1], 0.5)


def test_pairwise_identity_is_symmetric() -> None:
    seqs = ["ACDEFG", "ACDXXX", "AFFFFF"]
    mat = pairwise_identity(seqs)
    assert np.allclose(mat, mat.T)


def test_pairwise_identity_shorter_length() -> None:
    mat = pairwise_identity(["AAA", "AAAAA"])
    assert np.isclose(mat[0, 1], 1.0)  # all of shorter matches


def test_mean_pairwise_identity_independent_of_diagonal() -> None:
    """Mean off-diagonal only; diagonal 1s don't leak into the headline number."""
    mpi = mean_pairwise_identity(["ACDE", "FGHI", "KLMN"])
    assert 0.0 <= mpi <= 1.0
    assert mpi == 0.0  # no overlap between these disjoint quartets


def test_mean_pairwise_identity_requires_two() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        mean_pairwise_identity(["ACDE"])


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


def test_novelty_all_memorised_returns_ones() -> None:
    train = ["ACDE", "FGHI"]
    samples = ["ACDE", "FGHI"]
    out = max_identity_to_train(samples, train)
    assert np.allclose(out, 1.0)


def test_novelty_no_match_is_zero() -> None:
    train = ["AAAA", "BBBB"]
    samples = ["CCCC", "DDDD"]
    out = max_identity_to_train(samples, train)
    assert np.allclose(out, 0.0)


def test_novelty_partial_match() -> None:
    train = ["ACXXXX"]
    samples = ["ACDEEE"]
    out = max_identity_to_train(samples, train)
    # identity over 6 positions: 2/6 ~ 0.333
    assert np.isclose(out[0], 2.0 / 6.0)


def test_novelty_rejects_empty_train() -> None:
    with pytest.raises(ValueError):
        max_identity_to_train(["ACDE"], [])


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def test_held_out_loss_is_finite_and_positive(
    tokenizer: ProteinTokenizer, rng: np.random.Generator
) -> None:
    process = BFNProcess(vocab_size=tokenizer.vocab_size, beta_1=3.0)
    model = TimeConditionedTransformer(
        vocab_size=tokenizer.vocab_size,
        input_dim=tokenizer.vocab_size,
        max_length=LENGTH,
        dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        time_embed_dim=32,
    )
    seqs = generate_synthetic_proteins(rng, 16, min_length=16, max_length=28)
    ds = TokenizedDataset(seqs, tokenizer, max_length=LENGTH)
    batches = list(
        iterate_batches(
            ds, batch_size=BATCH, rng=rng, bucketed=False, fixed_length=LENGTH
        )
    )
    # dummy init
    ids, mask = batches[0]
    state = process.corrupt(rng, ids, np.full((ids.shape[0],), 0.5, dtype=np.float32), mask=mask)
    ts = init_train_state(
        model,
        state.tensor,
        state.t,
        mask,
        optax.sgd(0.0),
        key=jax.random.PRNGKey(0),
    )

    loss = compute_held_out_loss(
        model, ts.params, process, batches, rng=rng, n_time_samples=2
    )
    assert np.isfinite(loss)
    assert loss > 0.0


def test_held_out_loss_on_empty_batches_is_nan(
    tokenizer: ProteinTokenizer, rng: np.random.Generator
) -> None:
    process = BFNProcess(vocab_size=tokenizer.vocab_size, beta_1=3.0)
    model = TimeConditionedTransformer(
        vocab_size=tokenizer.vocab_size,
        input_dim=tokenizer.vocab_size,
        max_length=LENGTH,
        dim=16,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        time_embed_dim=16,
    )
    # Initialise the model to get valid params
    dummy_x = jnp.zeros((1, LENGTH, VOCAB), dtype=jnp.float32)
    dummy_t = jnp.zeros((1,), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, LENGTH), dtype=bool)
    params = model.init(jax.random.PRNGKey(0), dummy_x, dummy_t, dummy_mask)

    loss = compute_held_out_loss(model, params, process, [], rng=rng)
    assert np.isnan(loss)


# ---------------------------------------------------------------------------
# Pareto sweep
# ---------------------------------------------------------------------------


def test_pareto_sweep_returns_one_row_per_nfe(
    tokenizer: ProteinTokenizer, rng: np.random.Generator
) -> None:
    process = DFMProcess(vocab_size=tokenizer.vocab_size, mask_id=tokenizer.MASK_ID)
    model = TimeConditionedTransformer(
        vocab_size=tokenizer.vocab_size,
        input_dim=tokenizer.vocab_size,
        max_length=LENGTH,
        dim=16,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        time_embed_dim=16,
    )
    dummy_x = jnp.zeros((1, LENGTH, VOCAB), dtype=jnp.float32)
    dummy_t = jnp.zeros((1,), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, LENGTH), dtype=bool)
    params = model.init(jax.random.PRNGKey(0), dummy_x, dummy_t, dummy_mask)

    rows = pareto_sweep(
        process,
        model,
        params,
        tokenizer,
        rng=rng,
        batch_size=4,
        length=LENGTH,
        nfe_values=[4, 8, 16],
        metrics={"mean_identity": mean_pairwise_identity},
        n_batches=1,
    )
    assert len(rows) == 3
    for row in rows:
        assert "nfe" in row
        assert "mean_identity" in row
        assert 0.0 <= row["mean_identity"] <= 1.0


# ---------------------------------------------------------------------------
# ESM / ESMFold availability and error paths
# ---------------------------------------------------------------------------


def test_esm_availability_flag_is_bool() -> None:
    assert isinstance(esm.is_available(), bool)


def test_esmfold_availability_flag_is_bool() -> None:
    assert isinstance(esmfold.is_available(), bool)


def test_esm_raises_with_clear_error_if_unavailable() -> None:
    if esm.is_available():
        pytest.skip("torch + esm are installed; cannot assert unavailable path.")
    with pytest.raises(RuntimeError, match="fair-esm"):
        esm.compute_pseudo_likelihood(["ACDE"])


def test_esmfold_raises_with_clear_error_if_unavailable() -> None:
    if esmfold.is_available():
        pytest.skip("torch + esm are installed; cannot assert unavailable path.")
    with pytest.raises(RuntimeError, match="fair-esm"):
        esmfold.compute_foldability(["ACDE"])
