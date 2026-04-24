"""BFN JAX parity against InstaDeep's ``protein-sequence-bfn`` reference.

InstaDeep's public ``protein-sequence-bfn`` repository (Atkinson et al. 2024)
is the canonical reference for protein BFN training. This test, *if the
reference is installed*, verifies that our continuous-time loss agrees with
theirs to within ``1e-4`` on a fixed randomly generated batch.

The test is deliberately structured to be skipped -- not failed -- when the
reference is not importable. That way the rest of the test suite stays green
for contributors who only use this repo, while users who explicitly install
``protein-sequence-bfn`` alongside flowcompare get a strong correctness gate
against the published implementation.

Install pointer (see InstaDeep's repo README for up-to-date instructions):

    pip install git+https://github.com/instadeepai/protein-sequence-bfn.git

If InstaDeep change their module layout we will update the ``_import_reference``
helper below; failures with a ``SkipTest`` message are expected in that case.
"""

from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

from flowcompare.processes.bfn import BFNProcess, bfn_loss


def _reference_is_available() -> bool:
    """Only claim availability if the reference package actually exports the
    continuous-time loss we intend to compare against."""
    if importlib.util.find_spec("protein_sequence_bfn") is None:
        return False
    try:
        mod = importlib.import_module("protein_sequence_bfn")
    except ImportError:
        return False
    # We need some callable that computes the continuous-time categorical
    # BFN loss. The exact API has changed across versions; the two names we
    # have seen are listed here. Adjust as the reference evolves.
    return any(hasattr(mod, name) for name in ("categorical_loss", "discrete_loss"))


@pytest.mark.skipif(
    not _reference_is_available(),
    reason=(
        "InstaDeep protein-sequence-bfn not installed; parity test skipped. "
        "Install with:  pip install git+https://github.com/instadeepai/protein-sequence-bfn.git"
    ),
)
def test_bfn_loss_matches_instadeep_reference() -> None:
    """Our per-token continuous-time loss should agree with theirs on the
    same batch, same ``beta_1``, same ``(x_1, t, logits)``, to 1e-4."""
    import protein_sequence_bfn as ref

    batch, length, vocab = 4, 32, 25
    beta_1 = 3.0
    rng = np.random.default_rng(0)
    x1 = rng.integers(0, vocab, size=(batch, length), dtype=np.int32)
    t = rng.uniform(0.1, 0.9, size=(batch,)).astype(np.float32)
    logits = rng.standard_normal((batch, length, vocab)).astype(np.float32)

    ours = bfn_loss(logits, x1, t, vocab_size=vocab, beta_1=beta_1)

    # We intentionally avoid hard-coding the reference call signature --
    # each release of the upstream repo exposes a slightly different name.
    # The helper tries a few common ones; if none match, that is a parity
    # test failure worth investigating manually rather than a silent pass.
    ref_loss = None
    for candidate in ("categorical_loss", "discrete_loss"):
        fn = getattr(ref, candidate, None)
        if fn is None:
            continue
        try:
            ref_loss = np.asarray(
                fn(logits=logits, x1=x1, t=t, vocab_size=vocab, beta_1=beta_1)
            )
            break
        except TypeError:
            continue
    assert ref_loss is not None, (
        "protein_sequence_bfn is importable but none of the expected loss "
        "entrypoints accept our call signature; update this test."
    )

    # Compare per-token arrays. Tolerance of 1e-4 in absolute terms captures
    # float32 numerics; if parity is truly there, the error should be << 1e-5.
    assert ours.shape == ref_loss.shape
    assert np.allclose(ours, ref_loss, atol=1e-4), (
        f"max abs diff {np.max(np.abs(ours - ref_loss))} exceeds tolerance."
    )


# ---------------------------------------------------------------------------
# A self-consistency version of the parity test that always runs.
# ---------------------------------------------------------------------------


def test_bfn_loss_self_consistency_under_relabel() -> None:
    """Permutation of the vocabulary must permute predictions but preserve
    the BFN loss value: the loss depends only on the per-position L2 distance
    between ``e(x_1)`` and the network's softmax output.
    """
    batch, length, vocab = 4, 16, 10
    beta_1 = 3.0
    rng = np.random.default_rng(0)
    x1 = rng.integers(0, vocab, size=(batch, length), dtype=np.int32)
    t = rng.uniform(0.1, 0.9, size=(batch,)).astype(np.float32)
    logits = rng.standard_normal((batch, length, vocab)).astype(np.float32)
    l0 = bfn_loss(logits, x1, t, vocab_size=vocab, beta_1=beta_1)

    perm = rng.permutation(vocab)
    inv = np.argsort(perm)
    x1_perm = perm[x1]
    logits_perm = logits[..., inv]  # so softmax(logits_perm)[...,i] == softmax(logits)[...,perm[i]]
    l1 = bfn_loss(
        logits_perm, x1_perm, t, vocab_size=vocab, beta_1=beta_1
    )
    assert np.allclose(l0, l1, atol=1e-5)


def test_bfn_process_class_and_functional_helper_agree() -> None:
    """Same invariant as in test_bfn_math.py, repeated here so the parity
    file is a single place a contributor can look at for "does the BFN loss
    do the right thing"."""
    process = BFNProcess(vocab_size=12, beta_1=2.0)
    rng = np.random.default_rng(0)
    x1 = rng.integers(0, 12, size=(3, 8), dtype=np.int32)
    t = rng.uniform(0.0, 1.0, size=(3,)).astype(np.float32)
    state = process.corrupt(rng, x1, t)
    logits = rng.standard_normal((3, 8, 12)).astype(np.float32)
    class_loss = process.loss(logits, x1, state)
    fn_loss = bfn_loss(logits, x1, state.t, vocab_size=12, beta_1=2.0)
    assert np.allclose(class_loss, fn_loss, atol=1e-6)
