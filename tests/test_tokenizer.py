"""Unit tests for ProteinTokenizer."""

from __future__ import annotations

import numpy as np
import pytest

from flowcompare.tokenizer import STANDARD_AMINO_ACIDS, ProteinTokenizer


@pytest.fixture
def tok() -> ProteinTokenizer:
    return ProteinTokenizer()


def test_vocab_size(tok: ProteinTokenizer) -> None:
    assert tok.vocab_size == 25
    assert len(tok.amino_acids) == 20


def test_special_ids_distinct(tok: ProteinTokenizer) -> None:
    ids = {tok.PAD_ID, tok.BOS_ID, tok.EOS_ID, tok.MASK_ID, tok.UNK_ID}
    assert len(ids) == 5
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_encode_adds_bos_eos_by_default(tok: ProteinTokenizer) -> None:
    ids = tok.encode("ACD")
    assert ids.dtype == np.int32
    assert ids[0] == tok.BOS_ID
    assert ids[-1] == tok.EOS_ID
    assert ids.shape[0] == 5


def test_encode_no_special_tokens(tok: ProteinTokenizer) -> None:
    ids = tok.encode("ACD", add_special_tokens=False)
    assert ids.shape[0] == 3
    assert tok.BOS_ID not in ids.tolist()
    assert tok.EOS_ID not in ids.tolist()


def test_roundtrip_all_standard_amino_acids(tok: ProteinTokenizer) -> None:
    original = STANDARD_AMINO_ACIDS
    ids = tok.encode(original)
    decoded = tok.decode(ids)
    assert decoded == original


def test_encode_lowercase_normalised(tok: ProteinTokenizer) -> None:
    assert tok.decode(tok.encode("acd")) == "ACD"


def test_unknown_residues_map_to_unk(tok: ProteinTokenizer) -> None:
    ids = tok.encode("AXZ", add_special_tokens=False)
    assert ids.tolist() == [tok._aa_to_id["A"], tok.UNK_ID, tok.UNK_ID]


def test_decode_can_show_specials(tok: ProteinTokenizer) -> None:
    ids = tok.encode("AC")
    text = tok.decode(ids, skip_special=False)
    assert text.startswith("<bos>")
    assert text.endswith("<eos>")
    assert "AC" in text


def test_decode_rejects_out_of_range_ids(tok: ProteinTokenizer) -> None:
    with pytest.raises(ValueError):
        tok.decode([999])


def test_encode_batch_pads_to_longest(tok: ProteinTokenizer) -> None:
    ids, mask = tok.encode_batch(["A", "ACDE"])
    assert ids.shape == (2, 6)  # longest: BOS + ACDE + EOS = 6
    assert mask.dtype == bool
    assert mask[0].sum() == 3  # BOS + A + EOS
    assert mask[1].sum() == 6
    assert (ids[~mask] == tok.PAD_ID).all()


def test_encode_batch_truncates_and_preserves_eos(tok: ProteinTokenizer) -> None:
    long_seq = "A" * 50
    ids, mask = tok.encode_batch([long_seq], max_length=10)
    assert ids.shape == (1, 10)
    assert ids[0, 0] == tok.BOS_ID
    assert ids[0, -1] == tok.EOS_ID  # EOS preserved at truncation boundary
    assert mask[0].all()


def test_encode_batch_max_length_pads_short(tok: ProteinTokenizer) -> None:
    ids, mask = tok.encode_batch(["AC"], max_length=20)
    assert ids.shape == (1, 20)
    assert mask[0].sum() == 4  # BOS + AC + EOS
    assert (ids[0, 4:] == tok.PAD_ID).all()
