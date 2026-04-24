"""Protein sequence tokenizer.

Framework-agnostic: returns numpy arrays. Works with JAX (``jnp.asarray``) and
PyTorch (``torch.from_numpy``) callers without changes.

Vocabulary:
    0:  PAD   - padding
    1:  BOS   - beginning of sequence
    2:  EOS   - end of sequence
    3:  MASK  - absorbing/denoising mask
    4:  UNK   - non-standard residue (X, B, Z, J, U, O, ...)
    5-24: standard 20 amino acids in the canonical ACDEFGHIKLMNPQRSTVWY order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class ProteinTokenizer:
    """Tokenizer for the 20 standard amino acids plus 5 special tokens."""

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    MASK_ID: int = 3
    UNK_ID: int = 4
    _SPECIAL_IDS: tuple[int, ...] = (PAD_ID, BOS_ID, EOS_ID, MASK_ID, UNK_ID)
    _NUM_SPECIAL: int = 5

    def __init__(self) -> None:
        self._aa = STANDARD_AMINO_ACIDS
        self._aa_to_id: dict[str, int] = {
            aa: i + self._NUM_SPECIAL for i, aa in enumerate(self._aa)
        }
        # Reverse map covers specials as "" in position so decode() can skip them.
        self._id_to_aa: list[str] = (
            ["", "", "", "", ""] + list(self._aa)  # specials stringify to empty
        )
        self._special_tokens: dict[int, str] = {
            self.PAD_ID: "<pad>",
            self.BOS_ID: "<bos>",
            self.EOS_ID: "<eos>",
            self.MASK_ID: "<mask>",
            self.UNK_ID: "<unk>",
        }

    @property
    def vocab_size(self) -> int:
        return self._NUM_SPECIAL + len(self._aa)

    @property
    def amino_acids(self) -> str:
        return self._aa

    def encode(
        self,
        seq: str,
        *,
        add_special_tokens: bool = True,
    ) -> np.ndarray:
        """Encode a single sequence string to an int32 array of token ids."""
        seq = seq.upper().strip()
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.BOS_ID)
        for ch in seq:
            ids.append(self._aa_to_id.get(ch, self.UNK_ID))
        if add_special_tokens:
            ids.append(self.EOS_ID)
        return np.asarray(ids, dtype=np.int32)

    def decode(
        self,
        ids: Sequence[int] | np.ndarray,
        *,
        skip_special: bool = True,
    ) -> str:
        """Decode token ids back to an amino-acid string.

        When ``skip_special`` is False, special tokens are rendered with their
        angle-bracket names (``<bos>`` etc.) for debugging.
        """
        out: list[str] = []
        for raw in np.asarray(ids).tolist():
            tid = int(raw)
            if tid in self._SPECIAL_IDS:
                if skip_special:
                    continue
                out.append(self._special_tokens[tid])
            elif 0 <= tid < self.vocab_size:
                out.append(self._id_to_aa[tid])
            else:
                raise ValueError(f"Token id {tid} outside vocabulary [0, {self.vocab_size}).")
        return "".join(out)

    def encode_batch(
        self,
        seqs: Sequence[str],
        *,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode a batch and pad to a common length.

        Returns
        -------
        ids : int32 array of shape ``(batch, length)`` padded with ``PAD_ID``.
        mask : bool array of shape ``(batch, length)``; True on real tokens, False on padding.

        If ``max_length`` is given, sequences longer than it are truncated (with
        EOS preserved when ``add_special_tokens`` is True) and shorter ones are
        padded. If omitted, length equals the longest encoded sequence in the batch.
        """
        encoded = [self.encode(s, add_special_tokens=add_special_tokens) for s in seqs]

        if max_length is not None:
            truncated: list[np.ndarray] = []
            for arr in encoded:
                if arr.shape[0] <= max_length:
                    truncated.append(arr)
                elif add_special_tokens:
                    # Preserve BOS at start and EOS at end.
                    kept = arr[: max_length - 1]
                    truncated.append(np.concatenate([kept, np.asarray([self.EOS_ID], np.int32)]))
                else:
                    truncated.append(arr[:max_length])
            encoded = truncated
            length = max_length
        else:
            length = max((a.shape[0] for a in encoded), default=0)

        batch = len(encoded)
        ids = np.full((batch, length), self.PAD_ID, dtype=np.int32)
        mask = np.zeros((batch, length), dtype=bool)
        for i, arr in enumerate(encoded):
            n = arr.shape[0]
            ids[i, :n] = arr
            mask[i, :n] = True
        return ids, mask
