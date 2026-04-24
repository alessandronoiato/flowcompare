"""End-to-end training tests: loss decreases for both BFN and DFM on tiny data.

These tests are deliberately small-scale — a 2-layer transformer on 32
sequences — so they run in seconds on CPU, but they do exercise the entire
pipeline: tokenize -> batch -> corrupt -> jitted train step -> optax update.
"""

from __future__ import annotations

import jax
import numpy as np
import optax
import pytest

from flowcompare.data.collate import TokenizedDataset, iterate_batches
from flowcompare.data.synthetic import generate_synthetic_proteins
from flowcompare.models.transformer import TimeConditionedTransformer
from flowcompare.processes.bfn import BFNProcess
from flowcompare.processes.dfm import DFMProcess
from flowcompare.processes.identity import IdentityProcess
from flowcompare.tokenizer import ProteinTokenizer
from flowcompare.training.schedules import sample_time
from flowcompare.training.trainer import (
    TrainState,
    cycle,
    init_train_state,
    make_eval_step,
    make_train_step,
    train,
)

# ---------------------------------------------------------------------------
# Time-sampler tests
# ---------------------------------------------------------------------------


def test_sample_time_uniform_in_range() -> None:
    t = sample_time(np.random.default_rng(0), 64)
    assert t.shape == (64,)
    assert t.dtype == np.float32
    assert (t >= 0.0).all()
    assert (t <= 1.0).all()


def test_sample_time_stratified_covers_bins() -> None:
    """Stratified sampling should place exactly one sample in each of batch_size bins."""
    n = 16
    t = sample_time(np.random.default_rng(0), n, scheme="stratified")
    bins = np.floor(t * n).clip(0, n - 1).astype(int)
    assert len(set(bins.tolist())) == n


def test_sample_time_respects_t_max() -> None:
    t = sample_time(np.random.default_rng(0), 32, t_max=0.9)
    assert (t <= 0.9 + 1e-6).all()


def test_sample_time_rejects_bad_scheme() -> None:
    with pytest.raises(ValueError, match="scheme"):
        sample_time(np.random.default_rng(0), 4, scheme="bogus")


# ---------------------------------------------------------------------------
# Fixtures for the end-to-end tests
# ---------------------------------------------------------------------------


def _tiny_model(vocab_size: int) -> TimeConditionedTransformer:
    return TimeConditionedTransformer(
        vocab_size=vocab_size,
        input_dim=vocab_size,
        max_length=64,
        dim=32,
        num_heads=4,
        depth=2,
        mlp_ratio=2.0,
        time_embed_dim=32,
    )


def _tiny_dataset(rng: np.random.Generator, tokenizer: ProteinTokenizer):
    seqs = generate_synthetic_proteins(rng, 32, min_length=15, max_length=30)
    return TokenizedDataset(seqs, tokenizer, max_length=48)


# ---------------------------------------------------------------------------
# Training step wiring
# ---------------------------------------------------------------------------


def test_init_train_state_shapes() -> None:
    rng = np.random.default_rng(0)
    tok = ProteinTokenizer()
    process = BFNProcess(vocab_size=tok.vocab_size)
    model = _tiny_model(tok.vocab_size)
    ds = _tiny_dataset(rng, tok)
    ids, mask = next(
        iter(iterate_batches(ds, batch_size=4, rng=rng, bucketed=False, fixed_length=48))
    )
    state0 = process.corrupt(rng, ids, sample_time(rng, ids.shape[0]), mask=mask)
    opt = optax.adam(1e-3)
    ts = init_train_state(
        model, state0.tensor, state0.t, mask, opt, key=jax.random.PRNGKey(0)
    )
    assert isinstance(ts, TrainState)
    assert int(ts.step) == 0
    # ema copy distinct object from params
    params_leaves = jax.tree_util.tree_leaves(ts.params)
    ema_leaves = jax.tree_util.tree_leaves(ts.ema_params)
    assert len(params_leaves) == len(ema_leaves)
    for p, e in zip(params_leaves, ema_leaves, strict=False):
        assert np.allclose(np.asarray(p), np.asarray(e))


def test_make_train_step_returns_jittable() -> None:
    """Basic smoketest: build a train_step and run it once."""
    rng = np.random.default_rng(1)
    tok = ProteinTokenizer()
    process = BFNProcess(vocab_size=tok.vocab_size)
    model = _tiny_model(tok.vocab_size)
    opt = optax.adam(1e-3)

    ds = _tiny_dataset(rng, tok)
    ids, mask = next(
        iter(iterate_batches(ds, batch_size=4, rng=rng, bucketed=False, fixed_length=48))
    )
    state0 = process.corrupt(rng, ids, sample_time(rng, ids.shape[0]), mask=mask)
    ts = init_train_state(
        model, state0.tensor, state0.t, mask, opt, key=jax.random.PRNGKey(0)
    )
    step = make_train_step(model, process, opt)
    new_ts, loss = step(ts, state0.tensor, state0.t, ids, mask)
    assert np.isfinite(float(loss))
    assert int(new_ts.step) == 1


# ---------------------------------------------------------------------------
# End-to-end: loss goes down
# ---------------------------------------------------------------------------


def _train_and_return_losses(
    process,
    model,
    *,
    n_steps: int,
    seed: int,
    lr: float,
    batch_size: int = 4,
    time_scheme: str = "uniform",
    t_max: float = 1.0,
    fixed_length: int = 48,
):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    tok = ProteinTokenizer()
    ds = _tiny_dataset(np.random.default_rng(seed + 1), tok)

    def batches():
        return iterate_batches(
            ds,
            batch_size=batch_size,
            rng=rng,
            bucketed=False,
            fixed_length=fixed_length,
        )

    def n_batches_gen():
        cycled = cycle(batches)
        for _ in range(n_steps):
            yield next(cycled)

    opt = optax.adam(lr)
    metrics = train(
        model,
        process,
        opt,
        n_batches_gen(),
        rng=rng,
        key=key,
        time_scheme=time_scheme,
        t_max=t_max,
    )
    return metrics.losses


def test_bfn_training_reduces_loss() -> None:
    """BFN: average loss over last window should be lower than first window."""
    tok = ProteinTokenizer()
    process = BFNProcess(vocab_size=tok.vocab_size, beta_1=3.0)
    model = _tiny_model(tok.vocab_size)
    losses = _train_and_return_losses(
        process, model, n_steps=80, seed=123, lr=5e-3
    )
    assert all(np.isfinite(losses))
    first = float(np.mean(losses[:10]))
    last = float(np.mean(losses[-10:]))
    assert last < first * 0.9, f"BFN loss did not decrease: {first} -> {last}"


def test_dfm_training_reduces_loss() -> None:
    """DFM: same check for the mask-interpolant loss."""
    tok = ProteinTokenizer()
    process = DFMProcess(vocab_size=tok.vocab_size, mask_id=tok.MASK_ID)
    model = _tiny_model(tok.vocab_size)
    losses = _train_and_return_losses(
        process,
        model,
        n_steps=80,
        seed=456,
        lr=5e-3,
        t_max=0.99,  # avoid singularity
    )
    assert all(np.isfinite(losses))
    first = float(np.mean(losses[:10]))
    last = float(np.mean(losses[-10:]))
    assert last < first * 0.9, f"DFM loss did not decrease: {first} -> {last}"


def test_identity_training_reduces_loss() -> None:
    """IdentityProcess: this is plain cross-entropy on revealed x_1 — the
    easiest of the three — so a very short run should suffice."""
    tok = ProteinTokenizer()
    process = IdentityProcess(vocab_size=tok.vocab_size)
    model = _tiny_model(tok.vocab_size)
    losses = _train_and_return_losses(
        process, model, n_steps=40, seed=7, lr=5e-3
    )
    first = float(np.mean(losses[:5]))
    last = float(np.mean(losses[-5:]))
    assert last < first * 0.9


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def test_eval_step_matches_training_formula() -> None:
    """make_eval_step and make_train_step compute the same scalar on the same inputs."""
    rng = np.random.default_rng(0)
    tok = ProteinTokenizer()
    process = BFNProcess(vocab_size=tok.vocab_size)
    model = _tiny_model(tok.vocab_size)
    opt = optax.sgd(0.0)  # no-op optimizer so params don't change
    ds = _tiny_dataset(rng, tok)
    ids, mask = next(
        iter(iterate_batches(ds, batch_size=4, rng=rng, bucketed=False, fixed_length=48))
    )
    t = sample_time(rng, ids.shape[0])
    st = process.corrupt(rng, ids, t, mask=mask)

    ts = init_train_state(model, st.tensor, st.t, mask, opt, key=jax.random.PRNGKey(0))
    train_step = make_train_step(model, process, opt)
    eval_step = make_eval_step(model, process)

    _new_ts, train_loss = train_step(ts, st.tensor, st.t, ids, mask)
    eval_loss = eval_step(ts.params, st.tensor, st.t, ids, mask)
    assert np.isclose(float(train_loss), float(eval_loss), rtol=1e-5)
