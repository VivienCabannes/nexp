
import argparse
import os

from nexp.config import (
    CHECK_DIR,
    EMAIL,
    LOG_DIR,
)
from nexp.utils import (
    get_unique_path,
    touch_file,
)


class TrainingFramework:
    """Abstract base class for training frameworks.

    Parameters
    ----------
    args: Arguments from parser instanciated with `nexp.parser`.
    """
    def __init__(self, args: argparse.Namespace):
        pass

    def launch_slurm(self):
        """Launch experiment on cluster through slurm.
        
        Write a bash file to run experiments, and lauch it with sbatch.
        """
        self.register_logger()
        launcher_path = self.log_dir / "launcher.sh"
        touch_file(launcher_path)
        touch_file(self.check_path)

        with open(launcher_path, 'w') as f:
            f.write(f"#!/bin/bash\n\n")

            f.write(f"# Logging configuration\n")
            f.write(f"#SBATCH --job-name={self.config.job_name}\n")
            f.write(f"#SBATCH --output={self.log_dir}/%j-%t.out\n")
            f.write(f"#SBATCH --error={self.log_dir}/%j-%t.err\n")
            f.write(f"#SBATCH --mail-type=END\n")
            f.write(f"#SBATCH --mail-user={EMAIL}\n\n")

            f.write(f"# Job specfications\n")
            f.write(f"#SBATCH --time={self.config.time}\n")
            f.write(f"#SBATCH --mem={self.config.mem}\n")
            f.write(f"#SBATCH --nodes={self.config.nodes}\n")
            f.write(f"#SBATCH --ntasks-per-node={self.config.ntasks_per_node}\n")
            f.write(f"#SBATCH --gpus-per-node={self.config.gpus_per_node}\n")
            f.write(f"#SBATCH --cpus-per-task={self.config.cpus_per_task}\n")
            f.write(f"#SBATCH --partition={self.config.partition}\n")
            if self.config.constraint is not None:
                f.write(f"#SBATCH --constraint={self.config.constraint}\n\n")

            f.write(f"# Environment and job\n")
            f.write(f"source /private/home/vivc/.bashrc\n")
            f.write(f"newdev\n")

            # Parse argument strings
            arg_string = ""
            for key, value in vars(self.config).items():
                if key in [
                    "local", "job_name", "nodes", "gpus_per_node", "ntasks_per_node",
                    "constraint", "partition", "time", "cpus_per_task", "mem"
                ]:
                    # Skip arguments related to the cluster
                    continue
                if isinstance(value, bool) and not value:
                    # Skip arguments for which `action=store_true` if `False`
                    continue
                if isinstance(key, str):
                    # Replace underscores by hyphens
                    key = key.replace("_", "-")
                arg_string += f"--{key} {value} "
            f.write(f"python {self.file_path} {arg_string} --autolaunch\n")

        os.system(f"sbatch {launcher_path}")
    
    def register_logger(self):
        """Register logging and checkpoints paths.
        
        TODO
        ----
        Save into a json file, the job number, the paths and the hyperparameters (may make a function in `parser.py`).
        """
        self.log_dir = get_unique_path(LOG_DIR / self.config.job_name)

        self.check_dir = get_unique_path(CHECK_DIR / self.config.job_name)
        self.check_path = self.check_dir / "checkpoint.pth.tar"
        self.bestcheck_path = self.check_dir / "model_best.pth.tar"

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
