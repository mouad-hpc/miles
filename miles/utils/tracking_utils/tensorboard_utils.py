import datetime
import logging
import os

from miles.utils.misc import SingletonMeta

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

logger = logging.getLogger(__name__)


class TensorboardAdapter(metaclass=SingletonMeta):
    writer = None

    def __init__(self, args):
        assert args.use_tensorboard, f"{args.use_tensorboard=}"
        tb_project_name = args.tb_project_name
        tb_experiment_name = args.tb_experiment_name
        if tb_project_name is not None or os.environ.get("TENSORBOARD_DIR", None):
            if tb_project_name is not None and tb_experiment_name is None:
                tb_experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.initialize(tb_project_name, tb_experiment_name)
        else:
            raise ValueError("tb_project_name and tb_experiment_name, or TENSORBOARD_DIR are required")

    def initialize(self, tb_project_name, tb_experiment_name):
        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{tb_project_name}/{tb_experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()
