"""flowcompare: unified JAX benchmark of BFN and Discrete Flow Matching for proteins."""

from flowcompare.processes import IdentityProcess, ProcessState, SequenceProcess
from flowcompare.tokenizer import ProteinTokenizer

__all__ = [
    "ProteinTokenizer",
    "SequenceProcess",
    "ProcessState",
    "IdentityProcess",
]
__version__ = "0.0.2"
