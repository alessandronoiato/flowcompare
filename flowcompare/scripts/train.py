"""Train a single flowcompare model from the command line.

Usage example::

    python -m flowcompare.scripts.train \
        --process bfn \
        --data synthetic --synthetic-n 1024 \
        --dim 128 --depth 4 --num-heads 4 \
        --batch-size 32 --steps 2000 --lr 3e-4 \
        --out-dir runs/bfn_synthetic

Defaults are scaled down so the CLI works for smoketests on CPU; override
``--dim``, ``--depth``, ``--steps`` to move toward research-grade scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
import optax

from flowcompare.checkpoint import save_checkpoint
from flowcompare.data.collate import iterate_batches
from flowcompare.scripts.common import (
    add_data_args,
    add_model_args,
    add_process_args,
    build_model_config_from_args,
    build_process_from_args,
    load_sequences_from_args,
    make_dataset,
)
from flowcompare.tokenizer import ProteinTokenizer
from flowcompare.training.trainer import cycle, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_process_args(parser)
    add_model_args(parser)
    add_data_args(parser)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--time-scheme", default="uniform", choices=("uniform", "stratified"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write params.msgpack and config.json.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args(argv)

    tokenizer = ProteinTokenizer()
    process_cfg = build_process_from_args(args, vocab_size=tokenizer.vocab_size)
    model_cfg = build_model_config_from_args(args, vocab_size=tokenizer.vocab_size)
    process = process_cfg.build()
    model = model_cfg.build()

    sequences = load_sequences_from_args(args)
    if not sequences:
        raise RuntimeError("no sequences loaded from the configured data source.")
    dataset = make_dataset(sequences, tokenizer, max_length=model_cfg.max_length)

    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)

    def batches():
        return iterate_batches(
            dataset,
            batch_size=args.batch_size,
            rng=rng,
            bucketed=True,
            fixed_length=model_cfg.max_length,
        )

    def finite_batches():
        cycled = cycle(batches)
        for _ in range(args.steps):
            yield next(cycled)

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.lr, weight_decay=1e-4),
    )

    def log(step: int, loss: float) -> None:
        print(f"step {step:>6d}  loss {loss:.4f}", flush=True)

    metrics = train(
        model,
        process,
        opt,
        finite_batches(),
        rng=rng,
        key=key,
        ema_decay=args.ema_decay,
        log_every=args.log_every,
        time_scheme=args.time_scheme,
        t_max=args.t_max,
        callback=log,
    )

    out = save_checkpoint(
        args.out_dir,
        metrics.state.params,
        {
            "process": process_cfg.to_dict(),
            "model": model_cfg.to_dict(),
            "training": {
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "ema_decay": args.ema_decay,
                "seed": args.seed,
                "final_loss": metrics.losses[-1],
                "mean_first10": float(np.mean(metrics.losses[:10])),
                "mean_last10": float(np.mean(metrics.losses[-10:])),
            },
        },
    )
    print(f"saved checkpoint to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
