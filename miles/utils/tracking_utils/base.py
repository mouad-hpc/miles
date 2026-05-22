"""
Shared tracking interface for experiment logging backends.

Each backend implements ``init / log / finish``, and :class:`TrackingManager` fans out
calls to every active backend.

To add a new backend:
--------------------
1. Subclass :class:`TrackingBackend`.
2. Register it in :data:`BACKEND_REGISTRY`.
3. Add a corresponding ``--use-<name>`` CLI flag in ``arguments.py``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class TrackingBackend(ABC):

    @abstractmethod
    def init(self, args, *, primary: bool = True, **kwargs) -> None: ...

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int, step_key: str) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


class WandbBackend(TrackingBackend):

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        from . import wandb_utils

        if primary:
            wandb_utils.init_wandb_primary(args, **kwargs)
        else:
            wandb_utils.init_wandb_secondary(args, **kwargs)

    def log(self, metrics: dict[str, Any], step: int, step_key: str) -> None:
        import wandb

        wandb.log(metrics)

    def finish(self) -> None:
        import wandb

        wandb.finish()


class TensorboardBackend(TrackingBackend):
    def __init__(self) -> None:
        self.adapter = None

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        from .tensorboard_utils import TensorboardAdapter

        self.adapter = TensorboardAdapter(args)

    def log(self, metrics: dict[str, Any], step: int, step_key: str) -> None:
        if self.adapter is not None:
            data = {k: v for k, v in metrics.items() if k != step_key}
            self.adapter.log(data=data, step=step)

    def finish(self) -> None:
        if self.adapter is not None:
            self.adapter.finish()


class MlflowBackend(TrackingBackend):

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        from . import mlflow_utils

        mlflow_utils.init_mlflow(args, primary=primary, **kwargs)

    def log(self, metrics: dict[str, Any], step: int, step_key: str) -> None:
        from . import mlflow_utils

        mlflow_utils.log_metrics(metrics, step=step)

    def finish(self) -> None:
        from . import mlflow_utils

        mlflow_utils.finish()


class PrometheusBackend(TrackingBackend):

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        from .prometheus_utils import init_prometheus

        init_prometheus(args, start_server=primary)

    def log(self, metrics: dict[str, Any], step: int, step_key: str) -> None:
        from .prometheus_utils import get_prometheus

        prom = get_prometheus()
        assert prom is not None, (
            "Prometheus collector is not initialized; ensure init_tracking(..., primary=...) ran on the "
            "driver and workers can resolve the miles_prometheus_collector Ray actor."
        )
        prom.update.remote(metrics)

    def finish(self) -> None:
        return


BACKEND_REGISTRY: dict[str, tuple[type[TrackingBackend], str]] = {
    "wandb": (WandbBackend, "use_wandb"),
    "tensorboard": (TensorboardBackend, "use_tensorboard"),
    "mlflow": (MlflowBackend, "use_mlflow"),
    "prometheus": (PrometheusBackend, "use_prometheus"),
}


class TrackingManager:

    def __init__(self) -> None:
        self.backends: list[TrackingBackend] = []

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        for name, (cls, flag) in BACKEND_REGISTRY.items():
            if getattr(args, flag, False):
                logger.info("Initialising tracking backend: %s", name)
                backend = cls()
                backend.init(args, primary=primary, **kwargs)
                self.backends.append(backend)

    def log(self, metrics: dict[str, Any], step: int, step_key: str) -> None:
        for backend in self.backends:
            backend.log(metrics, step=step, step_key=step_key)

    def finish(self) -> None:
        for backend in self.backends:
            try:
                backend.finish()
            except Exception:
                logger.exception(
                    "Error finishing tracking backend %s",
                    type(backend).__name__,
                )
        self.backends.clear()
