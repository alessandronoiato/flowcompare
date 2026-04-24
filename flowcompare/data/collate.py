"""Tokenisation and length-bucketed batching.

A ``TokenizedDataset`` holds pre-encoded sequences (int32 numpy arrays) and
supports random shuffling and iteration. ``iterate_batches`` yields padded
``(ids, mask)`` batches ready to feed to the shared Transformer backbone.

Length bucketing keeps per-batch padding waste bounded: the corpus is sorted
by length, partitioned into buckets of similar length, and sampling picks
whole buckets at random. The user can opt out (``bucketed=False``) for
pure-random batching.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from flowcompare.tokenizer import ProteinTokenizer


class TokenizedDataset:
    """In-memory container for tokenised protein sequences.

    Parameters
    ----------
    sequences :
        Iterable of raw amino-acid strings.
    tokenizer :
        ``ProteinTokenizer`` to apply.
    max_length :
        Sequences encoded longer than this are truncated with EOS preserved.
    add_special_tokens :
        Whether BOS/EOS are added during encoding.
    """

    def __init__(
        self,
        sequences: Sequence[str],
        tokenizer: ProteinTokenizer,
        *,
        max_length: int = 512,
        add_special_tokens: bool = True,
    ) -> None:
        if max_length <= 2:
            raise ValueError(f"max_length must exceed 2, got {max_length}.")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self._encoded: list[np.ndarray] = []
        for seq in sequences:
            arr = tokenizer.encode(seq, add_special_tokens=add_special_tokens)
            if arr.shape[0] > max_length:
                if add_special_tokens:
                    arr = np.concatenate(
                        [arr[: max_length - 1], np.asarray([tokenizer.EOS_ID], np.int32)]
                    )
                else:
                    arr = arr[:max_length]
            self._encoded.append(arr.astype(np.int32))

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, i: int) -> np.ndarray:
        return self._encoded[i]

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray([a.shape[0] for a in self._encoded], dtype=np.int32)


def iterate_batches(
    dataset: TokenizedDataset,
    *,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
    drop_last: bool = True,
    bucketed: bool = True,
    bucket_size: int | None = None,
    fixed_length: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(ids, mask)`` batches.

    ``ids`` is shape ``(B, L)`` int32, padded with ``PAD_ID``. ``mask`` is
    shape ``(B, L)`` bool, True on real tokens.

    Parameters
    ----------
    dataset :
        A ``TokenizedDataset``.
    batch_size :
        Sequences per batch.
    rng :
        Numpy generator used for shuffling decisions.
    shuffle :
        If False, iterate sequentially.
    drop_last :
        If True, incomplete final batch is dropped.
    bucketed :
        If True, sort by length and sample whole buckets to bound padding
        overhead. If False, random batching.
    bucket_size :
        Number of sequences in each length-bucket; defaults to
        ``batch_size * 32``. Irrelevant when ``bucketed=False``.
    fixed_length :
        If set, every yielded batch is padded to exactly this length. This
        avoids JIT recompilation of downstream training steps that are
        sensitive to static shapes. Batches whose longest sequence exceeds
        ``fixed_length`` trigger a ``ValueError``; this is intentional
        because silently truncating would violate the dataset contract.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    n = len(dataset)
    if n == 0:
        return
    pad_id = dataset.tokenizer.PAD_ID

    if bucketed:
        indices = _bucketed_indices(
            dataset.lengths,
            rng,
            bucket_size=bucket_size or (batch_size * 32),
            shuffle=shuffle,
        )
    else:
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if drop_last and len(batch_idx) < batch_size:
            break
        yield _pad_batch(
            [dataset[int(i)] for i in batch_idx],
            pad_id=pad_id,
            fixed_length=fixed_length,
        )


def _bucketed_indices(
    lengths: np.ndarray,
    rng: np.random.Generator,
    *,
    bucket_size: int,
    shuffle: bool,
) -> np.ndarray:
    """Produce an index permutation with length-local batching."""
    order = np.argsort(lengths, kind="stable")
    if not shuffle:
        return order
    # Partition sorted order into buckets; shuffle inside each bucket so
    # similar-length items stay together but order within a bucket is random.
    n = len(order)
    bucket_count = max(1, (n + bucket_size - 1) // bucket_size)
    bucket_edges = np.linspace(0, n, bucket_count + 1, dtype=int)
    buckets: list[np.ndarray] = []
    for b_start, b_end in zip(bucket_edges[:-1], bucket_edges[1:], strict=False):
        chunk = order[b_start:b_end].copy()
        rng.shuffle(chunk)
        buckets.append(chunk)
    rng.shuffle(buckets)  # shuffle the order of buckets themselves
    return np.concatenate(buckets)


def _pad_batch(
    seqs: list[np.ndarray],
    *,
    pad_id: int,
    fixed_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    batch_max = max((s.shape[0] for s in seqs), default=0)
    if fixed_length is None:
        target = batch_max
    else:
        if batch_max > fixed_length:
            raise ValueError(
                f"sequence of length {batch_max} exceeds fixed_length {fixed_length}."
            )
        target = fixed_length
    ids = np.full((len(seqs), target), pad_id, dtype=np.int32)
    mask = np.zeros((len(seqs), target), dtype=bool)
    for i, s in enumerate(seqs):
        n = s.shape[0]
        ids[i, :n] = s
        mask[i, :n] = True
    return ids, mask
