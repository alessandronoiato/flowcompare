"""Process-agnostic neural backbones.

A backbone accepts a continuous state tensor ``(B, L, D_in)``, a time scalar
per example ``(B,)``, and an optional padding mask ``(B, L)``, and returns
``x_1``-prediction logits of shape ``(B, L, V)``. The same backbone serves
both BFN (where ``D_in`` is the simplex dimension) and DFM (where ``D_in`` is
the one-hot vocabulary) so that head-to-head comparisons share architecture
and parameter count.
"""

from flowcompare.models.transformer import TimeConditionedTransformer

__all__ = ["TimeConditionedTransformer"]
