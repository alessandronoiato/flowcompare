"""End-to-end test of the CLI scripts: train -> sample -> eval in tmp dir."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flowcompare.checkpoint import load_checkpoint, save_checkpoint
from flowcompare.models.transformer import TimeConditionedTransformer
from flowcompare.scripts import eval as eval_script
from flowcompare.scripts import sample as sample_script
from flowcompare.scripts import train as train_script

# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load_roundtrip(tmp_path: Path) -> None:
    model = TimeConditionedTransformer(
        vocab_size=25, input_dim=25, max_length=16, dim=16, depth=1, num_heads=2,
        mlp_ratio=2.0, time_embed_dim=16,
    )
    x = jnp.zeros((1, 16, 25), jnp.float32)
    t = jnp.zeros((1,), jnp.float32)
    mask = jnp.ones((1, 16), bool)
    params = model.init(jax.random.PRNGKey(0), x, t, mask)
    config = {"dim": 16, "depth": 1, "description": "unit-test model"}

    ckpt = save_checkpoint(tmp_path / "ck", params, config)
    assert (ckpt / "params.msgpack").exists()
    assert (ckpt / "config.json").exists()

    template = model.init(jax.random.PRNGKey(123), x, t, mask)  # different init
    loaded_params, loaded_config = load_checkpoint(ckpt, template)

    # Same shape; values match the originals.
    leaves_a = jax.tree_util.tree_leaves(params)
    leaves_b = jax.tree_util.tree_leaves(loaded_params)
    for a, b in zip(leaves_a, leaves_b, strict=False):
        assert np.allclose(np.asarray(a), np.asarray(b))
    assert loaded_config == config


# ---------------------------------------------------------------------------
# Full CLI flow: train then sample then eval
# ---------------------------------------------------------------------------


def _tiny_train_argv(out_dir: Path, process: str) -> list[str]:
    return [
        "--process", process,
        "--data", "synthetic",
        "--synthetic-n", "64",
        "--synthetic-min-length", "16",
        "--synthetic-max-length", "24",
        "--dim", "16",
        "--depth", "1",
        "--num-heads", "2",
        "--max-length", "32",
        "--mlp-ratio", "2.0",
        "--time-embed-dim", "16",
        "--batch-size", "4",
        "--steps", "20",
        "--lr", "5e-3",
        "--seed", "0",
        "--log-every", "100",
        "--out-dir", str(out_dir),
    ]


@pytest.mark.parametrize("process", ["bfn", "dfm", "identity"])
def test_end_to_end_train_sample_eval(tmp_path: Path, process: str, capsys) -> None:
    out = tmp_path / f"ckpt_{process}"
    rc = train_script.main(_tiny_train_argv(out, process))
    assert rc == 0
    assert (out / "params.msgpack").exists()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["process"]["name"] == process

    # sample: write fasta and verify content looks like protein strings
    fasta = tmp_path / f"samples_{process}.fasta"
    rc = sample_script.main(
        [
            "--ckpt", str(out),
            "--n", "4",
            "--length", "24",
            "--n-steps", "8",
            "--seed", "1",
            "--out", str(fasta),
        ]
    )
    assert rc == 0
    text = fasta.read_text()
    assert text.count(">sample_") == 4
    # Every sample line contains only standard amino acids after stripping.
    for line in text.splitlines():
        if not line.startswith(">"):
            for ch in line:
                assert ch in "ACDEFGHIKLMNPQRSTVWY"

    # eval: write JSON report
    report_path = tmp_path / f"report_{process}.json"
    rc = eval_script.main(
        [
            "--ckpt", str(out),
            "--eval-data", "synthetic",
            "--eval-n", "16",
            "--eval-min-length", "16",
            "--eval-max-length", "24",
            "--n-samples", "4",
            "--sample-length", "24",
            "--nfe", "4", "8",
            "--batch-size", "4",
            "--n-time-samples", "2",
            "--out", str(report_path),
        ]
    )
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["process"] == process
    assert "held_out_loss" in report
    assert "nfe_sweep" in report
    assert len(report["nfe_sweep"]) == 2
