"""
Configuration file to set up paths.
"""
import os
from pathlib import Path


user = os.environ['USER']
HOME = Path('/private/home/') / user

# Local paths
DATA_DIR = HOME / 'data'
SAVE_DIR = HOME / 'savings'
LOG_DIR = HOME / 'logs'

# Cluster paths
CHECK_DIR = Path(f'/checkpoint/{user}')

cifar10_path = "/datasets01/cifar-pytorch/11222017/"
cifar100_path = '/datasets01/cifar100/022818/data'
# imagenet_path = "/datasets01/imagenet_full_size/061417/train"
# tinyimagenet_path = "/datasets01/tiny-imagenet-200/train"

# Email notification
EMAIL = "vivc@meta.com"
