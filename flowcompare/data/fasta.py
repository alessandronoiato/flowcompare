"""Minimal dependency-free FASTA streaming reader.

Intentionally tiny: we do not need biotite or Biopython for this project's
data sizes, and avoiding them keeps the dependency footprint clean for users
who only want to train a model.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import IO

PathLike = str | os.PathLike[str] | IO[str]


def iter_fasta(source: PathLike) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` tuples from a FASTA source.

    Parameters
    ----------
    source :
        Either a filesystem path or an already-open text file object.

    Notes
    -----
    - Headers are yielded without the leading ``">"`` and without trailing
      whitespace.
    - Sequences are concatenated across wrapped lines; whitespace is stripped.
    - The last record is yielded at EOF even if the file does not end with a
      newline.
    """
    close_after = False
    if hasattr(source, "read"):
        stream = source  # type: ignore[assignment]
    else:
        stream = open(source, encoding="utf-8")  # noqa: SIM115 — streaming; closed in finally
        close_after = True

    try:
        header: str | None = None
        chunks: list[str] = []
        for line in stream:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)
    finally:
        if close_after:
            stream.close()


def count_fasta_records(source: PathLike) -> int:
    """Count records in a FASTA source without loading sequences into memory."""
    return sum(1 for _ in iter_fasta(source))
