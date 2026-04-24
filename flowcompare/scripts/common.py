"""Argument-parsing utilities shared by the CLI scripts.

Keeps train / sample / eval consistent: they all accept the same set of
``--process`` choices, the same model-size flags, and identical data
specification, so a training run and its downstream sampling run are easy
to align.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from flowcompare.data.collate import TokenizedDataset
from flowcompare.data.fasta import iter_fasta
from flowcompare.data.synthetic import generate_synthetic_proteins
from flowcompare.models.transformer import TimeConditionedTransformer
from flowcompare.processes.base import SequenceProcess
from flowcompare.processes.bfn import BFNProcess
from flowcompare.processes.dfm import DFMProcess
from flowcompare.processes.identity import IdentityProcess
from flowcompare.tokenizer import ProteinTokenizer

PROCESS_CHOICES = ("bfn", "dfm", "identity")


def add_process_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--process",
        choices=PROCESS_CHOICES,
        required=True,
        help="Which generative process to use.",
    )
    parser.add_argument(
        "--beta-1",
        type=float,
        default=3.0,
        help="BFN terminal accuracy (ignored for DFM/identity).",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--time-embed-dim", type=int, default=128)


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=str,
        default="synthetic",
        help="Either 'synthetic' or a path to a FASTA file.",
    )
    parser.add_argument("--synthetic-n", type=int, default=512)
    parser.add_argument("--synthetic-min-length", type=int, default=30)
    parser.add_argument("--synthetic-max-length", type=int, default=80)
    parser.add_argument("--data-seed", type=int, default=0)


@dataclass
class ProcessConfig:
    """Everything we need to reconstruct a process at sample/eval time."""

    name: str
    vocab_size: int
    beta_1: float
    mask_id: int

    def build(self) -> SequenceProcess:
        if self.name == "bfn":
            return BFNProcess(vocab_size=self.vocab_size, beta_1=self.beta_1)
        if self.name == "dfm":
            return DFMProcess(vocab_size=self.vocab_size, mask_id=self.mask_id)
        if self.name == "identity":
            return IdentityProcess(vocab_size=self.vocab_size)
        raise ValueError(f"unknown process {self.name!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "beta_1": self.beta_1,
            "mask_id": self.mask_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProcessConfig:
        return cls(
            name=d["name"],
            vocab_size=int(d["vocab_size"]),
            beta_1=float(d.get("beta_1", 3.0)),
            mask_id=int(d.get("mask_id", 3)),
        )


@dataclass
class ModelConfig:
    vocab_size: int
    input_dim: int
    max_length: int
    dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    time_embed_dim: int

    def build(self) -> TimeConditionedTransformer:
        return TimeConditionedTransformer(
            vocab_size=self.vocab_size,
            input_dim=self.input_dim,
            max_length=self.max_length,
            dim=self.dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            time_embed_dim=self.time_embed_dim,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        return cls(**d)


def build_process_from_args(args: argparse.Namespace, vocab_size: int) -> ProcessConfig:
    return ProcessConfig(
        name=args.process,
        vocab_size=vocab_size,
        beta_1=float(args.beta_1),
        mask_id=ProteinTokenizer.MASK_ID,
    )


def build_model_config_from_args(
    args: argparse.Namespace, vocab_size: int
) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        input_dim=vocab_size,
        max_length=int(args.max_length),
        dim=int(args.dim),
        depth=int(args.depth),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        time_embed_dim=int(args.time_embed_dim),
    )


def load_sequences_from_args(args: argparse.Namespace) -> list[str]:
    if args.data == "synthetic":
        rng = np.random.default_rng(args.data_seed)
        return generate_synthetic_proteins(
            rng,
            args.synthetic_n,
            min_length=args.synthetic_min_length,
            max_length=args.synthetic_max_length,
        )
    return [seq for _header, seq in iter_fasta(args.data)]


def make_dataset(
    sequences: list[str], tokenizer: ProteinTokenizer, max_length: int
) -> TokenizedDataset:
    return TokenizedDataset(sequences, tokenizer, max_length=max_length)
