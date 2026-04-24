"""Generative processes over protein sequences.

A ``SequenceProcess`` is a continuous-time interpolation between a simple
prior ``q_0`` and the data distribution ``q_1``. Both Bayesian Flow Networks
and Discrete Flow Matching instantiate this interface; they differ only in
what the intermediate state ``z_t`` is and how it evolves.
"""

from flowcompare.processes.base import ProcessState, SequenceProcess
from flowcompare.processes.bfn import BFNProcess, bfn_loss
from flowcompare.processes.dfm import DFMProcess, dfm_mask_loss
from flowcompare.processes.identity import IdentityProcess

__all__ = [
    "ProcessState",
    "SequenceProcess",
    "IdentityProcess",
    "BFNProcess",
    "bfn_loss",
    "DFMProcess",
    "dfm_mask_loss",
]
