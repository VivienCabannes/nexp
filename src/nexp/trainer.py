
import argparse
import logging
import os
from pathlib import Path

import torch

from nexp.config import (
    CHECK_DIR,
    LOG_DIR,
)
from nexp.launcher import SlurmLauncher
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
    
    def launch_slurm(self):
        """Launch the training on a SLURM cluster."""
        self.register_logger()
        launcher = SlurmLauncher(self.file_path, self.log_dir, self.config)
        launcher()
    
    def register_logger(self):
        """Register logging and checkpoints paths.
        
        TODO
        ----
        - Save into a json file, the job number, the paths and the hyperparameters (may make a function in `parser.py`).
        - Allow saving and loading from different files.
        """
        logger.debug("Registering loggers")
        self.log_dir = get_unique_path(LOG_DIR / self.config.job_name)

        self.check_dir = get_unique_path(CHECK_DIR / self.config.job_name)
        self.check_path = self.check_dir / "checkpoint.pth"
        self.bestcheck_path = self.check_dir / "model_best.pth"

    def save_checkpoint(self, full: bool = False, epoch: int = 0, file_path: str = None):
        """Save checkpoint.
        
        Parameters
        ----------
        full: Either to save optimizer state or not.
        epoch: Number of epoch in training.
        file_path: Path to save checkpoint to.
        TODO
        ----
        - Add accuracy to easily see if one beat best model by training more.
        """
        if full:
            state = {
                'epoch': epoch,
                'arch': self.config.architecture,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }
        else:
            state = {
                'arch': self.config.architecture,
                'state_dict': self.model.state_dict(),
            }
        if file_path is None:
            file_path = self.check_path
        torch.save(state, file_path)

    def load_checkpoint(self):
        """Load checkpoint.
        
        Parameters
        ----------
        file_name: path of the file to load checkpoint from.
        """
        if self.config.checkpoint:
            load_path = CHECK_DIR / self.config.checkpoint
        else:
            logging.debug("no checkpoint specified, training from scratch")
            return
        if not load_path.is_file():
            logging.warning(f"no checkpoint found at {str(load_path)}, training from scratch")
            return

        logging.info(f"loading checkpoint from {load_path}")
        checkpoint = torch.load(load_path, map_location="cpu")

        logging.debug(f"loading model weight")
        self.model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            logging.debug(f"loading optimizer state")
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logging.debug(f"loading scheduler state")
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            logging.debug(f"no optimizer state found in checkpoint")

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
