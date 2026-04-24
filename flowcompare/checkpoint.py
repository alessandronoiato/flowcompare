"""Checkpoint save/load for flowcompare.

Stores a bundle of ``(params, config_dict)`` to a single file using flax's
msgpack serialiser for the params and JSON for the config. Keeping config
with the checkpoint is important: different processes (BFN vs DFM) produce
different states and need to be paired with the correct network head at
load time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import flax.serialization
import numpy as np


def save_checkpoint(path: str | Path, params: Any, config: dict) -> Path:
    """Write ``params`` and ``config`` to ``path`` (a directory).

    Writes two files:
    - ``params.msgpack`` (flax serialisation of the pytree)
    - ``config.json``
    Creates ``path`` if it does not exist. Returns the final ``Path``.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    blob = flax.serialization.to_bytes(params)
    (out / "params.msgpack").write_bytes(blob)
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=_json_fallback)
    return out


def load_checkpoint(path: str | Path, template_params: Any) -> tuple[Any, dict]:
    """Load ``params`` and ``config`` from a directory saved by :func:`save_checkpoint`.

    ``template_params`` is required because flax's serialisation is
    structure-preserving: we need a pytree of the right shape to deserialise
    into. Typically this is obtained by re-initialising the model with a
    dummy input.
    """
    p = Path(path)
    blob = (p / "params.msgpack").read_bytes()
    params = flax.serialization.from_bytes(template_params, blob)
    with (p / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    return params, config


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")
