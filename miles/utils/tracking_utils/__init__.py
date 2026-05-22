import logging

from .base import TrackingManager

logger = logging.getLogger(__name__)
manager = TrackingManager()


def init_tracking(args, primary: bool = True, **kwargs):
    manager.init(args, primary=primary, **kwargs)


def log(args, metrics, step_key: str):
    step = metrics[step_key]
    manager.log(metrics, step=step, step_key=step_key)


def finish_tracking():
    manager.finish()
