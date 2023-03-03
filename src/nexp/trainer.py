
import argparse
import logging
import os

import torch

from nexp.config import (
    CHECK_DIR,
    EMAIL,
    LOG_DIR,
)
from nexp.utils import (
    get_unique_path,
    touch_file,
)
logger = logging.getLogger(__name__)


class Trainer:
    """Abstract base class for training frameworks.

    Parameters
    ----------
    args: Arguments from parser instanciated with `nexp.parser`.
    """
    def __init__(self, args: argparse.Namespace):
        pass
    
    def register_logger(self):
        """Register logging and checkpoints paths.
        
        TODO
        ----
        Save into a json file, the job number, the paths and the hyperparameters (may make a function in `parser.py`).
        """
        logger.info("Registering loggers")
        self.log_dir = get_unique_path(LOG_DIR / self.config.job_name)

        self.check_dir = get_unique_path(CHECK_DIR / self.config.job_name)
        self.check_path = self.check_dir / "checkpoint.pth"
        self.bestcheck_path = self.check_dir / "model_best.pth"

    @staticmethod
    def save_checkpoint(file_name, state):
        """Save checkpoint.
        
        Parameters
        ----------
        file_name: str
            Name of the file to save.
        state: dict
            Dictionary containing the state of the model.

        TODO
        ----
        Deal with multinode training.
        """
        torch.save(state, file_name)

    def register_architecture(self):
        """Register neural network architecture."""
        raise NotImplementedError

    def register_dataloader(self):
        """Register dataloader for training and validation."""
        raise NotImplementedError

    def register_optimizer(self):
        """Register optimizer."""
        raise NotImplementedError
    
    def __call__(self):
        raise NotImplementedError
