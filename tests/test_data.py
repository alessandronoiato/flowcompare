"""Tests for the data module: synthetic generator, FASTA reader, collator."""

from __future__ import annotations

import io

import numpy as np
import pytest

from flowcompare.data.collate import TokenizedDataset, iterate_batches
from flowcompare.data.fasta import count_fasta_records, iter_fasta
from flowcompare.data.synthetic import generate_synthetic_proteins
from flowcompare.tokenizer import STANDARD_AMINO_ACIDS, ProteinTokenizer


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def tokenizer() -> ProteinTokenizer:
    return ProteinTokenizer()


def test_synthetic_generator_shape_and_alphabet(rng: np.random.Generator) -> None:
    seqs = generate_synthetic_proteins(rng, 16, min_length=10, max_length=30)
    assert len(seqs) == 16
    for s in seqs:
        assert 10 <= len(s) <= 30
        assert all(ch in STANDARD_AMINO_ACIDS for ch in s)


def test_synthetic_generator_reproducible() -> None:
    a = generate_synthetic_proteins(np.random.default_rng(5), 8, min_length=20, max_length=20)
    b = generate_synthetic_proteins(np.random.default_rng(5), 8, min_length=20, max_length=20)
    assert a == b


def test_synthetic_rejects_bad_bounds(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError):
        generate_synthetic_proteins(rng, 4, min_length=50, max_length=10)


def test_synthetic_natural_distribution_is_non_uniform(
    rng: np.random.Generator,
) -> None:
    seqs = generate_synthetic_proteins(
        rng, 4, min_length=5000, max_length=5000, distribution="natural"
    )
    text = "".join(seqs)
    freqs = np.array([text.count(aa) for aa in STANDARD_AMINO_ACIDS], dtype=float)
    freqs = freqs / freqs.sum()
    # Should not look uniform.
    assert freqs.std() > 0.01


def test_iter_fasta_roundtrip() -> None:
    content = ">seq1 desc\nACDE\nFGHI\n>seq2\nKLMN\n"
    records = list(iter_fasta(io.StringIO(content)))
    assert records == [("seq1 desc", "ACDEFGHI"), ("seq2", "KLMN")]


def test_iter_fasta_handles_no_trailing_newline() -> None:
    content = ">only\nACD"
    records = list(iter_fasta(io.StringIO(content)))
    assert records == [("only", "ACD")]


def test_iter_fasta_skips_empty_lines() -> None:
    content = "\n\n>s1\n\nAC\n\nDE\n"
    records = list(iter_fasta(io.StringIO(content)))
    assert records == [("s1", "ACDE")]


def test_iter_fasta_empty_input_yields_nothing() -> None:
    assert list(iter_fasta(io.StringIO(""))) == []


def test_fasta_from_file(tmp_path) -> None:
    p = tmp_path / "x.fasta"
    p.write_text(">a\nACDE\n>b\nFGHIKL\n")
    assert list(iter_fasta(p)) == [("a", "ACDE"), ("b", "FGHIKL")]
    assert count_fasta_records(p) == 2


def test_tokenized_dataset_length_and_getitem(tokenizer: ProteinTokenizer) -> None:
    ds = TokenizedDataset(["AC", "DEFG"], tokenizer, max_length=32)
    assert len(ds) == 2
    assert ds[0][0] == tokenizer.BOS_ID
    assert ds[0][-1] == tokenizer.EOS_ID


def test_tokenized_dataset_truncates_long(tokenizer: ProteinTokenizer) -> None:
    ds = TokenizedDataset(["A" * 100], tokenizer, max_length=10)
    arr = ds[0]
    assert arr.shape[0] == 10
    assert arr[-1] == tokenizer.EOS_ID


def test_iterate_batches_basic_shapes(
    tokenizer: ProteinTokenizer, rng: np.random.Generator
) -> None:
    seqs = generate_synthetic_proteins(rng, 20, min_length=10, max_length=40)
    ds = TokenizedDataset(seqs, tokenizer, max_length=64)
    batches = list(iterate_batches(ds, batch_size=4, rng=rng, bucketed=False))
    assert len(batches) == 5
    for ids, mask in batches:
        assert ids.dtype == np.int32
        assert mask.dtype == bool
        assert ids.shape == mask.shape
        assert ids.shape[0] == 4
        # Padding positions must be PAD_ID, real positions must not be.
        assert (ids[~mask] == tokenizer.PAD_ID).all()


def test_iterate_batches_drop_last(
    tokenizer: ProteinTokenizer, rng: np.random.Generator
) -> None:
    seqs = generate_synthetic_proteins(rng, 10, min_length=5, max_length=10)
    ds = TokenizedDataset(seqs, tokenizer, max_length=32)
    full = list(iterate_batches(ds, batch_size=4, rng=rng, drop_last=True, bucketed=False))
    partial = list(
        iterate_batches(ds, batch_size=4, rng=rng, drop_last=False, bucketed=False)
    )
    assert len(full) == 2
    assert len(partial) == 3
    assert partial[-1][0].shape[0] == 2


def test_iterate_batches_bucketing_reduces_padding(
    tokenizer: ProteinTokenizer, rng: np.random.Generator
) -> None:
    seqs = generate_synthetic_proteins(rng, 64, min_length=10, max_length=60)
    ds = TokenizedDataset(seqs, tokenizer, max_length=80)

    def total_pad(batches) -> int:
        return sum(int((~m).sum()) for _, m in batches)

    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    unbucketed = list(iterate_batches(ds, batch_size=8, rng=rng_a, bucketed=False))
    bucketed = list(iterate_batches(ds, batch_size=8, rng=rng_b, bucketed=True))
    assert total_pad(bucketed) <= total_pad(unbucketed)


def test_iterate_batches_shuffle_determinism(
    tokenizer: ProteinTokenizer,
) -> None:
    seqs = generate_synthetic_proteins(
        np.random.default_rng(0), 16, min_length=10, max_length=10
    )
    ds = TokenizedDataset(seqs, tokenizer, max_length=32)
    b1 = list(iterate_batches(ds, batch_size=4, rng=np.random.default_rng(9), bucketed=False))
    b2 = list(iterate_batches(ds, batch_size=4, rng=np.random.default_rng(9), bucketed=False))
    for (a_ids, a_m), (b_ids, b_m) in zip(b1, b2, strict=False):
        assert np.array_equal(a_ids, b_ids)
        assert np.array_equal(a_m, b_m)


def test_iterate_batches_empty_dataset(tokenizer: ProteinTokenizer) -> None:
    ds = TokenizedDataset([], tokenizer, max_length=32)
    batches = list(iterate_batches(ds, batch_size=4, rng=np.random.default_rng(0)))
    assert batches == []


def test_bad_batch_size(tokenizer: ProteinTokenizer) -> None:
    ds = TokenizedDataset(["ACDE"], tokenizer, max_length=16)
    with pytest.raises(ValueError):
        list(iterate_batches(ds, batch_size=0, rng=np.random.default_rng(0)))
