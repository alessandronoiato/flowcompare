"""Generate sequences from a trained flowcompare checkpoint.

Usage example::

    python -m flowcompare.scripts.sample \
        --ckpt runs/bfn_synthetic \
        --n 64 --length 80 --n-steps 50 \
        --out samples.fasta
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from flowcompare.checkpoint import load_checkpoint
from flowcompare.sampling import sample_strings
from flowcompare.scripts.common import ModelConfig, ProcessConfig
from flowcompare.tokenizer import ProteinTokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--n", type=int, default=16, help="Number of samples.")
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="FASTA file to write; if omitted, prints to stdout.",
    )
    args = parser.parse_args(argv)

    tokenizer = ProteinTokenizer()
    # Peek the config to reconstruct process + model before loading params.
    import json

    with (args.ckpt / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    process_cfg = ProcessConfig.from_dict(cfg["process"])
    model_cfg = ModelConfig.from_dict(cfg["model"])
    process = process_cfg.build()
    model = model_cfg.build()

    dummy_x = jnp.zeros((1, model_cfg.max_length, model_cfg.vocab_size), jnp.float32)
    dummy_t = jnp.zeros((1,), jnp.float32)
    dummy_mask = jnp.ones((1, model_cfg.max_length), bool)
    template_params = model.init(jax.random.PRNGKey(0), dummy_x, dummy_t, dummy_mask)
    params, _ = load_checkpoint(args.ckpt, template_params)

    rng = np.random.default_rng(args.seed)
    strings = sample_strings(
        process,
        model.apply,
        params,
        tokenizer,
        rng=rng,
        batch_size=args.n,
        length=args.length,
        n_steps=args.n_steps,
    )

    def emit(stream) -> None:
        for i, s in enumerate(strings):
            stream.write(f">sample_{i:04d} process={process_cfg.name} nfe={args.n_steps}\n")
            stream.write(s + "\n")

    if args.out is None:
        import sys

        emit(sys.stdout)
    else:
        with args.out.open("w", encoding="utf-8") as f:
            emit(f)
        print(f"wrote {len(strings)} sequences to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
