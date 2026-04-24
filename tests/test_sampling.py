"""Tests for the process-agnostic sampling driver."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from flowcompare.models.transformer import TimeConditionedTransformer
from flowcompare.processes.bfn import BFNProcess
from flowcompare.processes.dfm import DFMProcess
from flowcompare.processes.identity import IdentityProcess
from flowcompare.sampling import (
    SamplingTrace,
    decode_state_to_tokens,
    nfe_sweep,
    sample_strings,
    sample_tokens,
)
from flowcompare.tokenizer import ProteinTokenizer

VOCAB = 25
BATCH = 2
LENGTH = 16


@pytest.fixture
def tokenizer() -> ProteinTokenizer:
    return ProteinTokenizer()


@pytest.fixture
def model() -> TimeConditionedTransformer:
    return TimeConditionedTransformer(
        vocab_size=VOCAB,
        input_dim=VOCAB,
        max_length=LENGTH,
        dim=16,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        time_embed_dim=16,
    )


@pytest.fixture
def params(model: TimeConditionedTransformer):
    import jax.numpy as jnp

    dummy_x = jnp.zeros((BATCH, LENGTH, VOCAB), dtype=jnp.float32)
    dummy_t = jnp.zeros((BATCH,), dtype=jnp.float32)
    dummy_mask = jnp.ones((BATCH, LENGTH), dtype=bool)
    return model.init(jax.random.PRNGKey(0), dummy_x, dummy_t, dummy_mask)


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# decode_state_to_tokens
# ---------------------------------------------------------------------------


def test_decode_prefers_extras_tokens() -> None:
    """When a state has extras['tokens'] (DFM), decoding uses it exactly."""
    dfm = DFMProcess(vocab_size=VOCAB, mask_id=3)
    rng = _rng()
    prior = dfm.sample_prior(rng, (BATCH, LENGTH))
    decoded = decode_state_to_tokens(prior, dfm)
    assert (decoded == 3).all()
    assert decoded.dtype == np.int32


def test_decode_argmax_when_no_extras() -> None:
    """BFN has no extras; decoding is argmax over simplex."""
    bfn = BFNProcess(vocab_size=VOCAB)
    rng = _rng()
    prior = bfn.sample_prior(rng, (BATCH, LENGTH))
    decoded = decode_state_to_tokens(prior, bfn)
    assert decoded.shape == (BATCH, LENGTH)
    assert decoded.dtype == np.int32


# ---------------------------------------------------------------------------
# sample_tokens shape and dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_process",
    [
        lambda: IdentityProcess(vocab_size=VOCAB),
        lambda: BFNProcess(vocab_size=VOCAB, beta_1=3.0),
        lambda: DFMProcess(vocab_size=VOCAB, mask_id=3),
    ],
    ids=["identity", "bfn", "dfm"],
)
def test_sample_tokens_shape_and_range(make_process, model, params) -> None:
    process = make_process()
    trace = sample_tokens(
        process,
        model.apply,
        params,
        rng=_rng(),
        batch_size=BATCH,
        length=LENGTH,
        n_steps=5,
    )
    assert isinstance(trace, SamplingTrace)
    assert trace.final_tokens.shape == (BATCH, LENGTH)
    assert trace.final_tokens.dtype == np.int32
    assert (trace.final_tokens >= 0).all()
    assert (trace.final_tokens < VOCAB).all()


def test_dfm_samples_contain_no_mask_tokens(model, params) -> None:
    """After a full sampling pass, the mask interpolant should have unmasked everything."""
    process = DFMProcess(vocab_size=VOCAB, mask_id=3)
    trace = sample_tokens(
        process,
        model.apply,
        params,
        rng=_rng(),
        batch_size=BATCH,
        length=LENGTH,
        n_steps=20,
    )
    assert (trace.final_tokens != 3).all()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_sample_tokens_is_deterministic_under_fixed_seed(model, params) -> None:
    process = BFNProcess(vocab_size=VOCAB, beta_1=3.0)
    a = sample_tokens(
        process,
        model.apply,
        params,
        rng=np.random.default_rng(42),
        batch_size=BATCH,
        length=LENGTH,
        n_steps=5,
    )
    b = sample_tokens(
        process,
        model.apply,
        params,
        rng=np.random.default_rng(42),
        batch_size=BATCH,
        length=LENGTH,
        n_steps=5,
    )
    assert np.array_equal(a.final_tokens, b.final_tokens)


def test_different_seeds_give_different_samples(model, params) -> None:
    process = BFNProcess(vocab_size=VOCAB, beta_1=3.0)
    a = sample_tokens(
        process, model.apply, params,
        rng=np.random.default_rng(1), batch_size=BATCH, length=LENGTH, n_steps=5,
    )
    b = sample_tokens(
        process, model.apply, params,
        rng=np.random.default_rng(2), batch_size=BATCH, length=LENGTH, n_steps=5,
    )
    # With random weights and only 5 steps, samples should overwhelmingly differ.
    assert not np.array_equal(a.final_tokens, b.final_tokens)


# ---------------------------------------------------------------------------
# Masking, padding
# ---------------------------------------------------------------------------


def test_sample_respects_padding_mask(model, params) -> None:
    process = BFNProcess(vocab_size=VOCAB, beta_1=3.0)
    mask = np.ones((BATCH, LENGTH), dtype=bool)
    mask[:, LENGTH - 4:] = False  # last 4 positions are pad
    trace = sample_tokens(
        process,
        model.apply,
        params,
        rng=_rng(),
        batch_size=BATCH,
        length=LENGTH,
        n_steps=5,
        mask=mask,
        pad_id=0,
    )
    assert (trace.final_tokens[:, LENGTH - 4:] == 0).all()


# ---------------------------------------------------------------------------
# sample_strings / decoding
# ---------------------------------------------------------------------------


def test_sample_strings_returns_decodable_sequences(
    tokenizer: ProteinTokenizer, model, params
) -> None:
    process = BFNProcess(vocab_size=tokenizer.vocab_size, beta_1=3.0)
    strs = sample_strings(
        process,
        model.apply,
        params,
        tokenizer,
        rng=_rng(),
        batch_size=BATCH,
        length=LENGTH,
        n_steps=4,
    )
    assert len(strs) == BATCH
    for s in strs:
        # decode with strip_special=True removes PAD/BOS/EOS/MASK/UNK; what's
        # left should only contain standard amino-acid characters.
        for ch in s:
            assert ch in "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# NFE sweep and recording
# ---------------------------------------------------------------------------


def test_record_trace_has_expected_length(model, params) -> None:
    process = BFNProcess(vocab_size=VOCAB, beta_1=3.0)
    trace = sample_tokens(
        process,
        model.apply,
        params,
        rng=_rng(),
        batch_size=BATCH,
        length=LENGTH,
        n_steps=7,
        record=True,
    )
    assert len(trace.tokens_per_step) == 7
    assert len(trace.t_per_step) == 7
    for snap in trace.tokens_per_step:
        assert snap.shape == (BATCH, LENGTH)


def test_nfe_sweep_returns_all_requested_points(model, params) -> None:
    process = BFNProcess(vocab_size=VOCAB, beta_1=3.0)
    out = nfe_sweep(
        process,
        model.apply,
        params,
        rng_factory=lambda: np.random.default_rng(0),
        batch_size=BATCH,
        length=LENGTH,
        n_steps_list=[2, 4, 8],
    )
    assert sorted(out.keys()) == [2, 4, 8]
    for tokens in out.values():
        assert tokens.shape == (BATCH, LENGTH)


def test_bad_n_steps_raises(model, params) -> None:
    process = BFNProcess(vocab_size=VOCAB, beta_1=3.0)
    with pytest.raises(ValueError, match="n_steps"):
        sample_tokens(
            process,
            model.apply,
            params,
            rng=_rng(),
            batch_size=BATCH,
            length=LENGTH,
            n_steps=0,
        )
