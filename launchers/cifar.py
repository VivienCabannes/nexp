
import argparse
import logging

import nexp.parser as nparser
from nexp.frameworks.cifar import CIFAR

logging.basicConfig(
    format="{asctime} {levelname} [{name:10s}:{lineno:3d}] {message}",
    style='{',
    datefmt='%H:%M:%S',
    level="INFO",
    handlers=[
        # Log to stderr, which is catched by SLURM into log files
        logging.StreamHandler(),
    ],
)

parser = argparse.ArgumentParser(
    description="Training configuration",
) 
nparser.decorate_parser(parser)
args = parser.parse_args()
nparser.fill_namespace(args)

framework = CIFAR(args)
framework()
