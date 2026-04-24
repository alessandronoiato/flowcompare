"""Time-conditioned Transformer backbone (Flax linen).

Minimal pre-norm Transformer that conditions on a continuous time ``t`` in
``[0, 1]`` via a sinusoidal embedding added to every position. The same
module is consumed by both BFN and DFM: the former feeds it simplex-valued
parameters, the latter feeds it one-hot tokens; both receive ``x_1``-prediction
logits back.

Design choices for this slice (intentionally minimal, room to grow):

- Learned absolute positional embeddings (upgrade to RoPE later if helpful).
- Time conditioning via broadcast addition after sinusoidal-then-MLP
  projection (upgrade to AdaLN/DiT-style if the gain shows up in ablations).
- No dropout yet; the loss landscapes of BFN and DFM will tell us whether
  regularisation is needed before we add it.
- Flax ``linen`` API rather than ``nnx`` for familiarity with the InstaDeep
  protein-sequence-bfn reference, which is written in a similar style.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from jax import Array


def sinusoidal_time_embedding(t: Array, dim: int) -> Array:
    """Standard sinusoidal embedding for a scalar time ``t`` per example.

    Parameters
    ----------
    t :
        Shape ``(batch,)`` float array in ``[0, 1]``.
    dim :
        Embedding dimension. Must be even for a clean sin/cos split.

    Returns
    -------
    Array of shape ``(batch, dim)``.
    """
    if dim % 2 != 0:
        raise ValueError(f"Time embedding dim must be even, got {dim}.")
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10_000.0) * jnp.arange(half) / max(half - 1, 1))
    args = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class FeedForward(nn.Module):
    dim: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x: Array) -> Array:
        hidden = int(self.dim * self.mlp_ratio)
        h = nn.Dense(hidden, name="fc1")(x)
        h = nn.gelu(h)
        h = nn.Dense(self.dim, name="fc2")(h)
        return h


class TransformerBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x: Array, attn_mask: Array | None) -> Array:
        h = nn.LayerNorm(name="ln_attn")(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            name="attn",
        )(h, h, mask=attn_mask)
        x = x + h

        h = nn.LayerNorm(name="ln_ffn")(x)
        h = FeedForward(dim=self.dim, mlp_ratio=self.mlp_ratio, name="ffn")(h)
        x = x + h
        return x


class TimeConditionedTransformer(nn.Module):
    """Pre-norm Transformer with additive sinusoidal time conditioning.

    Parameters
    ----------
    vocab_size :
        Number of output logit channels. Must match the ``SequenceProcess``
        vocabulary.
    input_dim :
        Expected channel count of the input continuous state. For BFN this
        equals ``vocab_size``; for DFM (one-hot path) it also equals
        ``vocab_size``. Kept explicit so that future process variants with
        richer state representations can set it separately.
    dim :
        Model width.
    depth :
        Number of transformer blocks.
    num_heads :
        Attention head count. Must divide ``dim``.
    max_length :
        Maximum sequence length supported by the learned positional table.
    mlp_ratio :
        FFN hidden-to-model width ratio.
    time_embed_dim :
        Dimension of the intermediate sinusoidal+MLP time embedding.

    Call signature
    --------------
    ``model(x, t, mask=None)``:
        - ``x``: ``(B, L, input_dim)`` float array.
        - ``t``: ``(B,)`` float array in ``[0, 1]``.
        - ``mask``: ``(B, L)`` bool array; True marks real tokens, False marks
          padding. If omitted, all positions attend freely.
        - Returns logits of shape ``(B, L, vocab_size)``.
    """

    vocab_size: int
    input_dim: int
    dim: int = 128
    depth: int = 4
    num_heads: int = 4
    max_length: int = 512
    mlp_ratio: float = 4.0
    time_embed_dim: int = 128

    def setup(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim {self.dim} must be divisible by num_heads {self.num_heads}."
            )

    @nn.compact
    def __call__(
        self,
        x: Array,
        t: Array,
        mask: Array | None = None,
    ) -> Array:
        if x.ndim != 3:
            raise ValueError(f"x must be (B, L, D), got shape {x.shape}.")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"x last-dim {x.shape[-1]} != configured input_dim {self.input_dim}."
            )
        batch, length, _ = x.shape
        if length > self.max_length:
            raise ValueError(
                f"Sequence length {length} exceeds max_length {self.max_length}."
            )
        if t.shape != (batch,):
            raise ValueError(f"t must have shape (B,)=({batch},), got {t.shape}.")

        h = nn.Dense(self.dim, name="input_proj")(x)

        pos_table = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (self.max_length, self.dim),
        )
        h = h + pos_table[None, :length, :]

        t_sin = sinusoidal_time_embedding(t, self.time_embed_dim)
        t_emb = nn.Dense(self.time_embed_dim, name="time_mlp_1")(t_sin)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(self.dim, name="time_mlp_2")(t_emb)
        h = h + t_emb[:, None, :]

        attn_mask = None
        if mask is not None:
            if mask.shape != (batch, length):
                raise ValueError(
                    f"mask shape {mask.shape} must match (B, L)=({batch},{length})."
                )
            attn_mask = nn.make_attention_mask(mask, mask)

        for i in range(self.depth):
            h = TransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                name=f"block_{i}",
            )(h, attn_mask)

        h = nn.LayerNorm(name="final_ln")(h)
        logits = nn.Dense(self.vocab_size, name="output_proj")(h)
        return logits
