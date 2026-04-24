# flowcompare

A unified JAX implementation and benchmark of **Bayesian Flow Networks**
(BFN; Graves et al. 2023) and **Discrete Flow Matching** (DFM; Gat et al.
2024 / Campbell et al. 2024) for protein sequence generation.

Both methods define a continuous-time process between a simple prior and
the data distribution and train a network to predict clean tokens from a
noised intermediate. They disagree on *where the noise lives*:

- **BFN** noises the parameters of a categorical distribution (continuous
  latent state on the simplex, Gaussian-noised).
- **DFM** (mask interpolant) noises in token space via an absorbing-state
  continuous-time Markov chain (discrete latent state, jump dynamics).

Every downstream difference (loss form, sampling algorithm, NFE/quality
tradeoff, compatibility with inpainting) flows from that one choice. This
repo implements both behind a single `SequenceProcess` interface, trains
them on matched protein datasets, and benchmarks them head-to-head.

The theoretical contrast and the specific predictions the benchmark is
designed to test are written up in [`docs/theory.md`](docs/theory.md).

## Status

Working implementations of:

- `BFNProcess` — categorical BFN with quadratic accuracy schedule
  $\beta(t) = \beta_1 t^2$, continuous-time loss, and Bayesian-update
  sampler (Graves 2023 Algs. 9–10). Includes a parity-test hook against
  InstaDeep's `protein-sequence-bfn` reference, skipped when the reference
  is not installed.
- `DFMProcess` — mask-interpolant DFM with $\kappa(t) = t$, the
  $1/(1-t)$-weighted continuous-time CE loss, and a tau-leaping sampler.
  Uniform-interpolant DFM is documented as a future extension.
- `TimeConditionedTransformer` — the shared backbone both processes use.
- Training loop (jitted, optax + EMA), generic sampler, evaluation suite
  (held-out ELBO, diversity, novelty, NFE Pareto), checkpointing, and CLI
  scripts (`flowcompare-train`, `flowcompare-sample`, `flowcompare-eval`).

Research-grade training runs (UniRef50 + OAS, ~30–150M params) are the
next step; infrastructure is ready but the reported benchmark numbers
will be produced once the training jobs complete. See
[`docs/reproduction.md`](docs/reproduction.md) for the recipe.

## Framework

JAX / Flax throughout. Evaluation against PyTorch-only models (ESM-2,
ESMFold) is shelled out to a torch subprocess (see `flowcompare/eval/esm.py`
and `flowcompare/eval/esmfold.py`) so the core package stays torch-free.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Quick start

The CLI uses a synthetic protein generator by default, so you can see a
full train → sample → eval loop on CPU in about 10 seconds:

```bash
flowcompare-train \
    --process bfn \
    --data synthetic --synthetic-n 256 \
    --dim 64 --depth 2 --num-heads 4 \
    --batch-size 16 --steps 200 --lr 3e-3 \
    --out-dir runs/bfn_smoketest

flowcompare-sample \
    --ckpt runs/bfn_smoketest \
    --n 16 --length 64 --n-steps 32 \
    --out runs/bfn_smoketest/samples.fasta

flowcompare-eval \
    --ckpt runs/bfn_smoketest \
    --eval-data synthetic --eval-n 128 \
    --n-samples 32 --nfe 8 16 32 64 \
    --out runs/bfn_smoketest/report.json
```

Swap `--process bfn` for `--process dfm` to run the same flow with
Discrete Flow Matching (mask interpolant).

## Library use

Everything the CLI does is a thin wrapper over importable Python. The
minimal end-to-end example:

```python
import jax, numpy as np, optax
from flowcompare.tokenizer import ProteinTokenizer
from flowcompare.data.synthetic import generate_synthetic_proteins
from flowcompare.data.collate import TokenizedDataset, iterate_batches
from flowcompare.processes.bfn import BFNProcess
from flowcompare.models.transformer import TimeConditionedTransformer
from flowcompare.training.trainer import train, cycle
from flowcompare.sampling import sample_strings

tok = ProteinTokenizer()
process = BFNProcess(vocab_size=tok.vocab_size, beta_1=3.0)
model = TimeConditionedTransformer(
    vocab_size=tok.vocab_size, input_dim=tok.vocab_size,
    max_length=128, dim=64, depth=2, num_heads=4,
    mlp_ratio=2.0, time_embed_dim=64,
)

rng = np.random.default_rng(0)
sequences = generate_synthetic_proteins(rng, 256, min_length=30, max_length=80)
dataset = TokenizedDataset(sequences, tok, max_length=128)

def batches():
    return iterate_batches(
        dataset, batch_size=16, rng=rng,
        bucketed=True, fixed_length=128,
    )

def finite():
    c = cycle(batches)
    for _ in range(200):
        yield next(c)

metrics = train(
    model, process, optax.adam(3e-3),
    finite(),
    rng=rng, key=jax.random.PRNGKey(0),
)

samples = sample_strings(
    process, model.apply, metrics.state.params, tok,
    rng=np.random.default_rng(7),
    batch_size=8, length=80, n_steps=32,
)
for s in samples:
    print(s)
```

## Layout

```
flowcompare/
  tokenizer.py           # 25-token protein tokenizer
  processes/             # SequenceProcess ABC + concrete implementations
    base.py              # ABC + ProcessState dataclass
    identity.py          # noise-free stub for infrastructure tests
    bfn.py               # categorical BFN (Graves 2023)
    dfm.py               # mask-interpolant DFM (Gat 2024 / Campbell 2024)
  models/
    transformer.py       # time-conditioned Transformer backbone
  data/
    synthetic.py         # random protein generator for CI / smoketests
    fasta.py             # dependency-free FASTA streaming reader
    collate.py           # tokenisation, bucketing, fixed-length padding
  training/
    schedules.py         # time samplers (uniform, stratified)
    trainer.py           # TrainState, jitted train_step, train/evaluate loops
  sampling.py            # process-agnostic sampler (NFE sweep helper)
  eval/
    perplexity.py        # held-out ELBO / NLL
    diversity.py         # pairwise identity
    novelty.py           # max identity to train corpus
    pareto.py            # NFE-vs-quality sweep
    esm.py esmfold.py    # torch-subprocess interfaces for ESM-2 / ESMFold
  checkpoint.py          # msgpack + JSON save/load
  scripts/               # flowcompare-train, -sample, -eval
docs/
  theory.md              # derivation, comparison, predictions
  reproduction.md        # benchmark recipe (UniRef50, OAS, GPU budget)
tests/                   # >= 100 tests, all deterministic
```

## Testing

```bash
pytest                 # runs the whole suite, ~20 s on CPU
pytest -m slow         # (none currently marked slow)
```

Every process is tested at three layers: (a) the `SequenceProcess` contract
(same 15-test suite parameterised over Identity / BFN / DFM), (b) the
process's own math invariants (e.g. "BFN's $\theta$ always sums to 1", "DFM
only unmasks, never re-masks"), and (c) an end-to-end training test that
asserts loss decreases on a 20-step run over synthetic data.

## License

MIT. See [`LICENSE`](LICENSE).

## Citing

If this repo is useful for your work, please cite the underlying methods
as well:

```bibtex
@article{graves2023bayesian,
  title={Bayesian Flow Networks},
  author={Graves, Alex and Srivastava, Rupesh Kumar and Atkinson, Timothy
          and Mandt, Stephan},
  journal={arXiv:2308.07037},
  year={2023}
}

@article{gat2024discrete,
  title={Discrete Flow Matching},
  author={Gat, Itai and Remez, Tal and Shaul, Neta and Kreis, Karsten
          and Chen, Ricky T. Q. and Synnaeve, Gabriel and Adi, Yossi
          and Lipman, Yaron},
  journal={arXiv:2407.15595},
  year={2024}
}
```
