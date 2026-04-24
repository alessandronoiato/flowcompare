"""Data loaders and collators for protein sequences.

Three layers:

- :mod:`flowcompare.data.synthetic` -- on-the-fly random protein generator used
  by smoketests and CI. Has no dependencies beyond numpy, so tests stay fast
  and deterministic under a seed.
- :mod:`flowcompare.data.fasta` -- streaming FASTA reader with no external
  dependency. Works for Pfam family dumps, UniRef subsets, and OAS chunks.
- :mod:`flowcompare.data.collate` -- tokenisation + length bucketing on top
  of ``ProteinTokenizer``.
"""

from flowcompare.data.collate import TokenizedDataset, iterate_batches
from flowcompare.data.fasta import iter_fasta
from flowcompare.data.synthetic import generate_synthetic_proteins

__all__ = [
    "iter_fasta",
    "generate_synthetic_proteins",
    "TokenizedDataset",
    "iterate_batches",
]
