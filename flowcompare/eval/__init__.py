"""Evaluation suite for flowcompare.

Five metrics, each in its own module:

1. :mod:`flowcompare.eval.perplexity` -- held-out token-level NLL / ELBO. For
   both BFN and DFM this is a Monte Carlo estimator of the continuous-time
   loss; we average over many ``t`` samples per sequence to reduce variance.
2. :mod:`flowcompare.eval.diversity` -- pairwise identity within a sample
   set.
3. :mod:`flowcompare.eval.novelty` -- max identity from each sample to the
   training corpus.
4. :mod:`flowcompare.eval.pareto` -- NFE sweep and helpers to build the
   quality/compute Pareto curve used in the theory doc.
5. :mod:`flowcompare.eval.esm` and :mod:`flowcompare.eval.esmfold` --
   interface to pytorch-only models via a subprocess; gated behind optional
   installs. These are the only evals that produce "bioscience credibility"
   numbers; they are deliberately separated so tests stay fast and CI stays
   green without heavyweight downloads.
"""

from flowcompare.eval.diversity import mean_pairwise_identity, pairwise_identity
from flowcompare.eval.novelty import max_identity_to_train
from flowcompare.eval.pareto import pareto_sweep
from flowcompare.eval.perplexity import compute_held_out_loss

__all__ = [
    "pairwise_identity",
    "mean_pairwise_identity",
    "max_identity_to_train",
    "pareto_sweep",
    "compute_held_out_loss",
]
