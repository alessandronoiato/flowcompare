"""Evaluate a flowcompare checkpoint: held-out loss, diversity, novelty,
and an NFE sweep.

Usage example::

    python -m flowcompare.scripts.eval \
        --ckpt runs/bfn_synthetic \
        --eval-data synthetic --eval-seed 7 \
        --n-samples 64 --nfe 8 16 32 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from flowcompare.checkpoint import load_checkpoint
from flowcompare.data.collate import TokenizedDataset, iterate_batches
from flowcompare.data.fasta import iter_fasta
from flowcompare.data.synthetic import generate_synthetic_proteins
from flowcompare.eval.diversity import mean_pairwise_identity
from flowcompare.eval.novelty import max_identity_to_train
from flowcompare.eval.pareto import pareto_sweep
from flowcompare.eval.perplexity import compute_held_out_loss
from flowcompare.sampling import sample_strings
from flowcompare.scripts.common import ModelConfig, ProcessConfig
from flowcompare.tokenizer import ProteinTokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--eval-data", type=str, default="synthetic")
    parser.add_argument("--eval-n", type=int, default=256)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-min-length", type=int, default=30)
    parser.add_argument("--eval-max-length", type=int, default=80)
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--sample-length", type=int, default=80)
    parser.add_argument("--nfe", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-time-samples", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    tokenizer = ProteinTokenizer()
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

    # Build eval dataset
    if args.eval_data == "synthetic":
        eval_rng = np.random.default_rng(args.eval_seed)
        eval_sequences = generate_synthetic_proteins(
            eval_rng,
            args.eval_n,
            min_length=args.eval_min_length,
            max_length=args.eval_max_length,
        )
    else:
        eval_sequences = [seq for _h, seq in iter_fasta(args.eval_data)]
    eval_ds = TokenizedDataset(
        eval_sequences, tokenizer, max_length=model_cfg.max_length
    )
    rng = np.random.default_rng(args.eval_seed + 1)
    batches = list(
        iterate_batches(
            eval_ds,
            batch_size=args.batch_size,
            rng=rng,
            bucketed=False,
            fixed_length=model_cfg.max_length,
        )
    )

    # Held-out loss
    held_out_loss = compute_held_out_loss(
        model,
        params,
        process,
        batches,
        rng=rng,
        n_time_samples=args.n_time_samples,
        t_max=0.99 if process_cfg.name == "dfm" else 1.0,
    )

    # Sample a batch at a fixed NFE (largest) for diversity / novelty
    samples = sample_strings(
        process,
        model.apply,
        params,
        tokenizer,
        rng=rng,
        batch_size=args.n_samples,
        length=args.sample_length,
        n_steps=max(args.nfe),
    )
    diversity = mean_pairwise_identity(samples) if len(samples) >= 2 else float("nan")
    novelty = float(np.mean(max_identity_to_train(samples, eval_sequences)))

    # Pareto sweep
    sweep = pareto_sweep(
        process,
        model,
        params,
        tokenizer,
        rng=rng,
        batch_size=args.n_samples,
        length=args.sample_length,
        nfe_values=args.nfe,
        metrics={"mean_identity": mean_pairwise_identity},
        n_batches=1,
    )

    report = {
        "process": process_cfg.name,
        "held_out_loss": held_out_loss,
        "held_out_perplexity": float(np.exp(held_out_loss)) if held_out_loss else None,
        "n_eval_sequences": len(eval_sequences),
        "n_samples": len(samples),
        "diversity_mean_pairwise_identity": diversity,
        "novelty_mean_max_identity_to_eval": novelty,
        "nfe_sweep": sweep,
    }

    text = json.dumps(report, indent=2, default=lambda o: float(o))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote evaluation report to {args.out}", flush=True)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
