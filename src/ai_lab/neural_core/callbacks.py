from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class Callback(Protocol):
    def on_train_begin(self, model: Any) -> None: ...

    def on_epoch_end(self, model: Any, logs: dict[str, float | int | None]) -> bool: ...

    def on_train_end(self, model: Any) -> None: ...


class BaseCallback:
    def on_train_begin(self, model: Any) -> None:
        del model

    def on_epoch_end(self, model: Any, logs: dict[str, float | int | None]) -> bool:
        del model, logs
        return False

    def on_train_end(self, model: Any) -> None:
        del model


@dataclass(slots=True)
class EarlyStopping(BaseCallback):
    monitor: str = "val_loss"
    patience: int = 10
    min_delta: float = 0.0
    restore_best_weights: bool = True
    fallback_monitor: str = "loss"
    best_value_: float = field(init=False, default=float("inf"))
    wait_: int = field(init=False, default=0)
    best_parameters_: list[dict[str, Any]] | None = field(init=False, default=None)
    stopped_epoch_: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be > 0.")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be >= 0.")

    def on_train_begin(self, model: Any) -> None:
        del model
        self.best_value_ = float("inf")
        self.wait_ = 0
        self.best_parameters_ = None
        self.stopped_epoch_ = None

    def on_epoch_end(self, model: Any, logs: dict[str, float | int | None]) -> bool:
        monitored = logs.get(self.monitor)
        if monitored is None:
            monitored = logs.get(self.fallback_monitor)
        if monitored is None:
            return False

        current = float(monitored)
        if current < self.best_value_ - self.min_delta:
            self.best_value_ = current
            self.wait_ = 0
            if self.restore_best_weights:
                self.best_parameters_ = model._copy_trainable_parameters()
            return False

        self.wait_ += 1
        if self.wait_ < self.patience:
            return False

        self.stopped_epoch_ = int(logs["epoch"]) if logs.get("epoch") is not None else None
        if self.restore_best_weights and self.best_parameters_ is not None:
            model._restore_trainable_parameters(self.best_parameters_)
        return True


@dataclass(slots=True)
class ModelCheckpoint(BaseCallback):
    path: str | Path
    monitor: str = "val_loss"
    save_best_only: bool = True
    fallback_monitor: str = "loss"
    best_value_: float = field(init=False, default=float("inf"))

    def on_train_begin(self, model: Any) -> None:
        del model
        self.best_value_ = float("inf")

    def on_epoch_end(self, model: Any, logs: dict[str, float | int | None]) -> bool:
        monitored = logs.get(self.monitor)
        if monitored is None:
            monitored = logs.get(self.fallback_monitor)
        if monitored is None:
            return False

        current = float(monitored)
        if not self.save_best_only or current < self.best_value_:
            self.best_value_ = current
            model.save(self.path)
        return False


@dataclass(slots=True)
class LearningRateScheduler(BaseCallback):
    schedule: Any

    def on_epoch_end(self, model: Any, logs: dict[str, float | int | None]) -> bool:
        epoch = int(logs["epoch"])
        new_rate = float(self.schedule(epoch, model.learning_rate))
        if new_rate <= 0.0:
            raise ValueError("Scheduled learning rate must be > 0.")
        model.learning_rate = new_rate
        model._optimizer.learning_rate = new_rate
        return False


@dataclass(slots=True)
class HistoryLogger(BaseCallback):
    records: list[dict[str, float | int | None]] = field(default_factory=list)

    def on_train_begin(self, model: Any) -> None:
        del model
        self.records.clear()

    def on_epoch_end(self, model: Any, logs: dict[str, float | int | None]) -> bool:
        del model
        self.records.append(dict(logs))
        return False
