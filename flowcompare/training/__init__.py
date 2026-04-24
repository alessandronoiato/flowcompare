"""Training loop, time samplers, optimizer wrappers, and EMA.

Exposes:

- :func:`flowcompare.training.schedules.sample_time` -- draws continuous time
  values in ``[0, 1]`` for the continuous-time estimators of both processes.
- :class:`flowcompare.training.trainer.TrainState` -- dataclass carrying
  ``params``, ``ema_params``, ``opt_state``, ``step``.
- :func:`flowcompare.training.trainer.make_train_step` -- returns a jitted
  single-step function parameterised by ``(model, process, optimizer)``.
- :func:`flowcompare.training.trainer.train` -- full training loop using an
  iterable of ``(ids, mask)`` batches.
"""

from flowcompare.training.schedules import sample_time
from flowcompare.training.trainer import (
    TrainState,
    evaluate,
    make_eval_step,
    make_train_step,
    train,
)

__all__ = [
    "sample_time",
    "TrainState",
    "make_train_step",
    "make_eval_step",
    "train",
    "evaluate",
]
