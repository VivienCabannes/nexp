"""
NB
--
Will probably be renamed to `cifar.py`.
"""
import os
import json
import pathlib
from typing import Tuple

import torchvision.datasets as datasets

from nexp.config import (
    DATA_DIR,
    cifar10_path,
    cifar100_path,
)
MEAN_STD_PATH = DATA_DIR / "stats_mean_std.json"


def get_mean_std(name: str, dataset: datasets = None, recompute: bool = False) -> Tuple[list[int], list[int]]:
    """
    Retrieve or compute mean and std of a dataset and save them in a json file.

    Parameters
    ----------
    name: name of the dataset
    dataset: dataset to compute the mean and std
    recompute: if True, recompute the mean and std

    Returns
    -------
    mean: mean of the dataset
    std: standard deviation of the dataset
    """
    path = MEAN_STD_PATH
    if not os.path.exists(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({}, f)
    with open(path, 'r') as f:
        datasets_stats = json.load(f)
    if not recompute and name in datasets_stats:
        mean = datasets_stats[name]['mean']
        std = datasets_stats[name]['std']
    else:
        if name in ['cifar10', 'cifar100']:
            mean = dataset.data.mean(axis=(0, 1, 2)) / 255
            std = dataset.data.std(axis=(0, 1, 2)) / 255
            datasets_stats[name] = {'mean': mean.tolist(), 'std': std.tolist()}
        else:
            raise NotImplementedError(f"Dataset {name} is not implemented.")
        with open(path, 'w') as f:
            json.dump(datasets_stats, f, indent=4)
    return mean, std


def compute_mean_std(name: str) -> None:
    """
    Compute mean and std of a dataset and save them in a json file.

    Parameters
    ----------
    name: name of the dataset as in `torchvision.datasets`
    """
    match name:
        case "cifar10":
            dataset = datasets.CIFAR10(root=cifar10_path, train=True, download=False)
        case "cifar100":
            dataset = datasets.CIFAR100(root=cifar100_path, train=True, download=False)
        case _:
            raise NotImplementedError(f"Dataset {name} is not implemented.")
    return get_mean_std(name, dataset, recompute=True)
