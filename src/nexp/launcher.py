
import argparse
import os
from pathlib import Path

from nexp.config import EMAIL
from nexp.parser import SLURM_ONLY
from nexp.utils import touch_file


class SlurmLauncher:
    """Launch experiment on cluster through slurm.

    Parameters
    ----------
    log_dir: Path to the directory where the experiment will be logged.
    config: Arguments from parser instanciated with `nexp.parser`.
    """
    def __init__(self, file_path: str, log_dir: str, config: argparse.Namespace):
        self.config = config
        self.file_path = Path(file_path)
        self.log_dir = Path(log_dir)

    def __call__(self):
        """Write a bash file to run experiments, and lauch it with sbatch."""
        launcher_path = self.log_dir / "launcher.sh"
        touch_file(launcher_path)

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
            f.write(f"#SBATCH --partition={self.config.partition}\n\n")
            if self.config.constraint is not None:
                f.write(f"#SBATCH --constraint={self.config.constraint}\n\n")

            f.write(f"# Environment and job\n")
            f.write(f"source /private/home/vivc/.bashrc\n")
            f.write(f"newdev\n")

            # Parse argument strings
            arg_string = self.recreate_args(self.config)
            f.write(f"python {self.file_path} {arg_string}\n")

        os.system(f"sbatch {launcher_path}")

    @staticmethod
    def recreate_args(config: argparse.Namespace) -> str:
        """Recreate command line arguments from config.
         
        Parameters
        ----------
        config: Namespace from parser instanciated with `nexp.parser`.
        """
        arg_string = ""
        for key, value in vars(config).items():
            if key in SLURM_ONLY:
                # Skip arguments related to the cluster
                continue
            if isinstance(value, bool) and not value:
                # Skip arguments for which `action=store_true` if `False`
                continue
            if isinstance(value, str) and not value:
                # Skip empty string argument
                continue
            if isinstance(value, int) and value == -1:
                # Integer arguments set to `-1` should be ignored
                continue
            if isinstance(key, str):
                # Replace underscores by hyphens
                key = key.replace("_", "-")
            arg_string += f"--{key} {value} "
        return arg_string
