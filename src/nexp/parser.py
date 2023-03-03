
import argparse
from datetime import datetime
from pathlib import Path
import os

from nexp.config import (
    CHECK_DIR,
    LOG_DIR,
)


# Useful variable for debugging
ctl = False

def decorate_parser(parser: argparse.ArgumentParser, training: bool = True) -> None:
    """Create parser with basic configuration.
    
    Parameters
    ----------
    parser: Parser to be decorated.
    training: Whether experiments are for training or inference.
    """
    architecture_config(parser)
    if training:
        optimizer_config(parser)
    else:
        raise NotImplementedError("Inference is not implemented yet.")
    cluster_config(parser)
    logger_config(parser)


def architecture_config(parser: argparse.ArgumentParser) -> None:
    # Setting details
    setting = parser.add_argument_group("Architecture")
    setting.add_argument(
        "-a", "--architecture", "-a", default="resnet18", type=str, 
        help="Neural network architecture", metavar="\b",
    )
    setting.add_argument("--float16", action="store_true")


def optimizer_config(parser: argparse.ArgumentParser) -> None:
    # Optimizer details
    optimizer = parser.add_argument_group("Optimizer")
    optimizer.add_argument(
        "-bs", "--batch-size", default=64, required=ctl, type=int,
        help="batch size (per GPU)", metavar="\b",
    )
    optimizer.add_argument(
        "--full-batch-size", default=64, type=int,
        help="batch size (across GPUs)", metavar="\b",
    )
    optimizer.add_argument(
        "-e", "--epochs", default=1, required=ctl, type=int,
        help="number of epochs", metavar="\b",
    )
    optimizer.add_argument(
        "--optimizer", type=str, 
        choices=["Adam", "AdamW", "SGD", "LARS", "RMSprop"], default="AdamW",
        help="optimizer", metavar="\b",
    )
    optimizer.add_argument(
        "-lr", "--learning-rate", default=.1, required=ctl, type=float,
        help="learning rate", metavar="\b",
    )
    optimizer.add_argument(
        "--momentum", default=0, type=float,
        help="momentum", metavar="\b",
    )
    optimizer.add_argument(
        "--weight-decay", default=0, type=float,
        help="weight decay", metavar="\b",
    )


def cluster_config(parser: argparse.ArgumentParser) -> None:
    # Cluster details
    cluster = parser.add_argument_group("Cluster settings")
    cluster.add_argument(
        "--local", action="store_true",
        help="run experiments locally",
    )
    cluster.add_argument(
        "-J", "--job-name", type=str, default="nexp",
        help="name of job", metavar="\b",
    )
    cluster.add_argument(
        "-N", "--nodes", type=int, default=1,
        help="number of nodes on which to run", metavar="\b",
    )
    cluster.add_argument(
        "-G", "--gpus-per-node", type=int, default=8,
        help="number of GPUs required per allocated node",  metavar="\b",
    )
    cluster.add_argument(
        "--ntasks-per-node", type=int, default=1,
        help="maximal number of tasks to invoke on each node", metavar="\b",
    )
    cluster.add_argument(
        "--constraint", type=list, default=None,
        help="specify a list of constraints", metavar="\b",
    )
    cluster.add_argument(
        "--partition", type=str, default="devlab",
        help="partition requested", metavar="\b",
    )
    cluster.add_argument(
        "--time", type=int, default="4320",
        help="time limit", metavar="\b",
    )
    cluster.add_argument(
        "--log", type=str, default=LOG_DIR,
        help="location of log redirection", metavar="\b",
    )


def logger_config(parser: argparse.ArgumentParser) -> None:
    logger = parser.add_argument_group("Logs")
    logger.add_argument(
        "--check-freq", type=int, default=1,
        help="checkpoint frequency in number of epochs", metavar="\b",
    )
    logger.add_argument(
        "--log-freq", default=10, type=int, 
        help="log frequency in number of epochs", metavar="\b",
    )
    logger.add_argument(
        "--check", default=CHECK_DIR, type=str,
        help="location of checkpoint", metavar="\b",
    )
    logger.add_argument(
        "--seed", type=int, default=-1, 
        help="seed to fix randomness", metavar="\b",
    )


def fill_namespace(args: argparse.Namespace) -> None:
    """Fill namespace with deterministic variables."""
    fill_cpus(args)


def fill_cpus(args: argparse.Namespace) -> None:
    """Compute CPUs number and RAM based on number of GPUs
    
    The function saved the result in `args.cpus_per_task`.
    According to online documentation:
        1. Requesting less than 10 CPUs per GPU will make the GPU efficiency low. 
    Requesting more than 10 CPUs will result in some GPUs not schedulable and thus GPU resource waste.
        2. All learnfair machines, including 8 GPU machines and 2 GPU machines have 512GB RAM per node.

    Parameters
    ----------
    args: Argument parser decorated through `cluster_config`.
    """
    num_gpus = args.gpus_per_node * args.nodes
    args.cpus_per_task = 8 * num_gpus
    args.mem = str(32 * num_gpus) + "G"
    args.full_batch_size = args.batch_size * num_gpus
