
from datetime import datetime
import os
from pathlib import Path
import random

import numpy as np
import torch


def set_all_seed(seed, fast=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if fast:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


## File system utils

def get_unique_path(file_path: str) -> Path:
    """Get a unique path by adding a day identifier.
    
    Parameters
    ----------
    file_path: Path to be checked.

    Returns
    -------
    path: Unique path.
    """
    day_id = datetime.utcnow().strftime("%Y%m%d")
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    path = file_path / day_id
    if os.path.exists(path):
        add_id = 1
        while True:
            add_id += 1
            if not os.path.exists(path / str(add_id)):
                path = path / str(add_id)
                break
    return path


def touch_file(file_path: str) -> None:
    """Create file if it does not exist.
    
    Parameters
    ----------
    file_path: Path to file.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    try:
        file_path.touch(exist_ok=True)
    except FileNotFoundError:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch(exist_ok=True)


def touch_dir(file_path: str) -> None:
    """Create directory if it does not exist.
    
    Parameters
    ----------
    file_path: Path to directory.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    file_path.mkdir(parents=True, exist_ok=True)


if __name__=="__main__":
    print("Hello world!")