# Reproduction

This document is a recipe for reproducing the benchmark numbers that
`flowcompare` is designed to produce: a head-to-head comparison of
Bayesian Flow Networks (BFN) and Discrete Flow Matching (DFM, mask
interpolant) on matched protein datasets at matched compute.

All claims in the paper / report that the benchmark will write are
reproducible from this document plus the code in `main`.

## 1. What is being measured

For each process $p \in \{\text{BFN}, \text{DFM}\}$, we train a single
time-conditioned Transformer (same architecture, same optimiser, same
data, same token budget) and report:

| axis              | metric                                                       |
| ----------------- | ------------------------------------------------------------ |
| likelihood        | held-out continuous-time loss (ELBO proxy), perplexity       |
| sequence quality  | ESM-2 pseudo-likelihood (`flowcompare/eval/esm.py`)          |
| structural quality| ESMFold pLDDT, pTM (`flowcompare/eval/esmfold.py`)           |
| diversity         | mean pairwise identity within samples                        |
| novelty           | max identity of each sample to the training set              |
| efficiency        | the full quality-vs-NFE Pareto curve                         |

The first, fourth, fifth, and sixth columns are fully contained in this
package (no torch). Columns 2–3 call out to a torch subprocess; the
interface stubs are in `flowcompare/eval/esm.py` and
`flowcompare/eval/esmfold.py`, and the worker scripts ship with ESM-2
and ESMFold checkpoints that the user must install separately.

## 2. Datasets

Two datasets are used, both in FASTA format.

### 2.1 UniRef50 subset (general protein pretraining)

- Source: [UniRef50](https://www.uniprot.org/help/uniref) release
  matching the target date of the report. The current release is fine;
  record the release tag in the config.
- Filter: keep sequences with length in `[40, 256]`, standard amino
  acids only (drop sequences containing `B, J, O, U, X, Z`).
- Subsample: draw a random 500K sequences with seed `0` and hold out 5K
  for validation / 5K for test.

Expected file layout:

```
data/uniref50/
  train.fasta      # 490,000 sequences
  val.fasta        #   5,000
  test.fasta       #   5,000
```

### 2.2 OAS antibody subset (domain-specific)

- Source: the [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/)
  paired (heavy + light) dump for humans.
- Filter: V-region only, length in `[90, 140]`, drop sequences with any
  non-standard residue.
- Subsample: 100K train / 2K val / 2K test.

```
data/oas/
  train.fasta
  val.fasta
  test.fasta
```

Why two datasets: the theory predicts that DFM's weaker consistency
under inpainting should hurt it more on OAS (where useful generation is
*conditional* on a CDR scaffold) than on UniRef50 (unconditional). See
`docs/theory.md` §6 for the precise prediction.

## 3. Training configurations

Both processes are trained with **identical** model, optimiser, and data
configs. The only change across runs is `--process {bfn|dfm}` and,
optionally, `--beta-1` for BFN.

### 3.1 Small (30M parameters) — for fast iteration

```bash
flowcompare-train \
    --process bfn \
    --data data/uniref50/train.fasta \
    --dim 384 --depth 6 --num-heads 6 \
    --max-length 256 --mlp-ratio 4.0 --time-embed-dim 384 \
    --batch-size 64 --steps 100000 --lr 3e-4 \
    --ema-decay 0.9999 --time-scheme stratified \
    --out-dir runs/bfn_small_uniref
```

Approx. compute budget: ~12h on a single A100 / H100.

### 3.2 Medium (150M parameters) — headline configuration

```bash
flowcompare-train \
    --process bfn \
    --data data/uniref50/train.fasta \
    --dim 768 --depth 12 --num-heads 12 \
    --max-length 256 --mlp-ratio 4.0 --time-embed-dim 768 \
    --batch-size 64 --steps 300000 --lr 2e-4 \
    --ema-decay 0.9999 --time-scheme stratified \
    --out-dir runs/bfn_medium_uniref
```

Approx. compute budget: ~3 days on 4xA100.

Repeat with `--process dfm` to produce the matched DFM run. The two
runs consume exactly the same total number of tokens.

### 3.3 Antibody runs

Repeat both configurations with `--data data/oas/train.fasta` and
`--out-dir runs/{bfn,dfm}_{small,medium}_oas`.

## 4. Evaluation

Once training completes, produce the benchmark numbers:

```bash
# Held-out ELBO / perplexity, diversity, novelty, and an NFE sweep.
flowcompare-eval \
    --ckpt runs/bfn_medium_uniref \
    --eval-data data/uniref50/test.fasta \
    --n-samples 512 --sample-length 256 \
    --nfe 8 16 32 64 128 256 \
    --batch-size 32 \
    --out runs/bfn_medium_uniref/report.json
```

Do the same with `--ckpt runs/dfm_medium_uniref`; the two reports are
the head-to-head.

### 4.1 ESM-2 pseudo-likelihood (optional, requires torch)

```bash
# Sample first, then score.
flowcompare-sample \
    --ckpt runs/bfn_medium_uniref --n 512 --length 256 --n-steps 128 \
    --out runs/bfn_medium_uniref/samples_nfe128.fasta

python -m flowcompare.eval.esm_worker \
    --fasta runs/bfn_medium_uniref/samples_nfe128.fasta \
    --model esm2_t33_650M_UR50D \
    --out runs/bfn_medium_uniref/esm_pll.json
```

The worker script is the torch-side counterpart to
`flowcompare/eval/esm.py`; it lives in a sibling repo (or can be written
against `facebookresearch/esm`) and is not shipped here to keep the JAX
tree torch-free.

### 4.2 ESMFold pLDDT / pTM (optional, requires torch + GPU)

Same pattern as ESM-2, wrapping an ESMFold worker. See
`flowcompare/eval/esmfold.py` for the expected input/output contract.

## 5. Pareto curves

The NFE sweep in step 4 produces per-NFE samples and per-NFE metrics.
Plot:

- x: NFE (= number of network evaluations)
- y: ESM-2 pseudo-likelihood (or ELBO if ESM is unavailable)
- one curve per process.

The theoretical prediction is that BFN dominates at low NFE (because
its update is a smooth Bayesian step rather than an on/off unmasking
event) and DFM catches up only with many steps. See `docs/theory.md` §6.

## 6. Smoketest (CPU, ~1 minute)

To confirm your install is healthy before the real runs:

```bash
# Trains a tiny model on 256 synthetic proteins, samples 16 sequences,
# evaluates them. No GPUs, no torch, no external data.
flowcompare-train \
    --process bfn --data synthetic --synthetic-n 256 \
    --dim 64 --depth 2 --num-heads 4 --max-length 96 \
    --batch-size 16 --steps 200 --lr 3e-3 \
    --out-dir /tmp/fc_smoketest

flowcompare-sample --ckpt /tmp/fc_smoketest --n 8 --length 64 --n-steps 32 \
    --out /tmp/fc_smoketest/samples.fasta

flowcompare-eval --ckpt /tmp/fc_smoketest --eval-data synthetic \
    --eval-n 128 --n-samples 16 --nfe 8 16 32 \
    --out /tmp/fc_smoketest/report.json

cat /tmp/fc_smoketest/report.json
```

`report.json` should contain finite numbers for `held_out_loss`,
`diversity_mean_pairwise_identity`, and `novelty_mean_max_identity_to_eval`,
and one entry per NFE in `nfe_sweep`.

## 7. Reporting

Each real run's `report.json` is the artifact quoted in the writeup. A
single table with rows `{BFN-small, DFM-small, BFN-medium, DFM-medium} ×
{UniRef50, OAS}` and columns `{perplexity, ESM-PLL, pLDDT, pTM,
diversity, novelty}` constitutes the headline result; the full Pareto
curves appear as figures.

Every number is reproducible from the corresponding `--out-dir` and
`--out` path using the seeds and hyperparameters recorded in
`config.json` / the CLI invocation above.
