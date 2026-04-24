"""Shape, determinism, gradient, and jit tests for the time-conditioned backbone."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowcompare.models.transformer import (
    TimeConditionedTransformer,
    sinusoidal_time_embedding,
)

VOCAB = 25
BATCH = 2
LENGTH = 9
DIM = 32
DEPTH = 2
HEADS = 4


@pytest.fixture
def model() -> TimeConditionedTransformer:
    return TimeConditionedTransformer(
        vocab_size=VOCAB,
        input_dim=VOCAB,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        max_length=64,
        time_embed_dim=DIM,
    )


@pytest.fixture
def inputs():
    x = np.random.default_rng(0).standard_normal((BATCH, LENGTH, VOCAB)).astype(np.float32)
    t = np.array([0.1, 0.8], dtype=np.float32)
    return jnp.asarray(x), jnp.asarray(t)


@pytest.fixture
def init_params(model: TimeConditionedTransformer, inputs):
    x, t = inputs
    return model.init(jax.random.PRNGKey(0), x, t)


def test_sinusoidal_time_embedding_shape_and_finite() -> None:
    t = jnp.linspace(0.0, 1.0, 5)
    emb = sinusoidal_time_embedding(t, 16)
    assert emb.shape == (5, 16)
    assert jnp.all(jnp.isfinite(emb))
    assert float(jnp.abs(emb).max()) <= 1.0 + 1e-6


def test_sinusoidal_time_embedding_rejects_odd_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        sinusoidal_time_embedding(jnp.zeros(1), 7)


def test_forward_shape(model, init_params, inputs) -> None:
    x, t = inputs
    logits = model.apply(init_params, x, t)
    assert logits.shape == (BATCH, LENGTH, VOCAB)
    assert logits.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(logits))


def test_forward_is_deterministic(model, init_params, inputs) -> None:
    x, t = inputs
    a = model.apply(init_params, x, t)
    b = model.apply(init_params, x, t)
    assert jnp.allclose(a, b)


def test_different_time_changes_output(model, init_params, inputs) -> None:
    x, _ = inputs
    t_early = jnp.full((BATCH,), 0.05, dtype=jnp.float32)
    t_late = jnp.full((BATCH,), 0.95, dtype=jnp.float32)
    a = model.apply(init_params, x, t_early)
    b = model.apply(init_params, x, t_late)
    assert not jnp.allclose(a, b, atol=1e-3)


def test_mask_changes_output(model, init_params, inputs) -> None:
    x, t = inputs
    no_mask = model.apply(init_params, x, t)
    mask = jnp.ones((BATCH, LENGTH), dtype=jnp.bool_)
    mask = mask.at[:, -3:].set(False)
    masked = model.apply(init_params, x, t, mask)
    assert not jnp.allclose(no_mask, masked)


def test_mask_shape_validation(model, init_params, inputs) -> None:
    x, t = inputs
    bad_mask = jnp.ones((BATCH, LENGTH + 1), dtype=jnp.bool_)
    with pytest.raises(ValueError, match="mask shape"):
        model.apply(init_params, x, t, bad_mask)


def test_input_dim_validation(model, init_params, inputs) -> None:
    _, t = inputs
    bad_x = jnp.zeros((BATCH, LENGTH, VOCAB + 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="input_dim"):
        model.apply(init_params, bad_x, t)


def test_length_exceeds_max_is_rejected() -> None:
    small = TimeConditionedTransformer(
        vocab_size=VOCAB, input_dim=VOCAB, dim=DIM, depth=1,
        num_heads=HEADS, max_length=4, time_embed_dim=DIM,
    )
    x = jnp.zeros((1, 8, VOCAB), dtype=jnp.float32)
    t = jnp.zeros((1,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="exceeds max_length"):
        small.init(jax.random.PRNGKey(0), x, t)


def test_head_divisibility_validated() -> None:
    bad = TimeConditionedTransformer(
        vocab_size=VOCAB, input_dim=VOCAB, dim=30, depth=1,
        num_heads=4, max_length=16, time_embed_dim=16,
    )
    x = jnp.zeros((1, 4, VOCAB), dtype=jnp.float32)
    t = jnp.zeros((1,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="divisible"):
        bad.init(jax.random.PRNGKey(0), x, t)


def test_jit_compiles_and_matches_eager(model, init_params, inputs) -> None:
    x, t = inputs
    eager = model.apply(init_params, x, t)
    jitted = jax.jit(lambda p, x, t: model.apply(p, x, t))(init_params, x, t)
    assert jnp.allclose(eager, jitted, atol=1e-5)


def test_gradients_flow_to_all_params(model, init_params, inputs) -> None:
    x, t = inputs
    target = jnp.zeros((BATCH, LENGTH), dtype=jnp.int32)

    def loss_fn(params):
        logits = model.apply(params, x, t)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        picked = jnp.take_along_axis(log_probs, target[..., None], axis=-1)[..., 0]
        return -picked.mean()

    loss, grads = jax.value_and_grad(loss_fn)(init_params)
    assert jnp.isfinite(loss)

    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    for g in leaves:
        assert jnp.all(jnp.isfinite(g))
    assert any(float(jnp.abs(g).max()) > 0 for g in leaves)


def test_parameter_count_is_reasonable(init_params) -> None:
    leaves = jax.tree_util.tree_leaves(init_params)
    total = sum(int(np.prod(leaf.shape)) for leaf in leaves)
    assert 1_000 < total < 500_000


def test_different_batch_size_shares_params(model, init_params) -> None:
    x4 = jnp.zeros((4, LENGTH, VOCAB), dtype=jnp.float32)
    t4 = jnp.zeros((4,), dtype=jnp.float32)
    out = model.apply(init_params, x4, t4)
    assert out.shape == (4, LENGTH, VOCAB)


def test_time_shape_validation(model, init_params, inputs) -> None:
    x, _ = inputs
    bad_t = jnp.zeros((BATCH + 1,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="t must have shape"):
        model.apply(init_params, x, bad_t)
