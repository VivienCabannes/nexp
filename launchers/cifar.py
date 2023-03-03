
import argparse

import nexp.parser as nparser
from nexp.frameworks.cifar import CIFAR

parser = argparse.ArgumentParser(
    description="Training configuration",
) 
nparser.decorate_parser(parser)
args = parser.parse_args()
nparser.fill_namespace(args)

framework = CIFAR(args)
if args.local:
    # Launch calculations locally 
    framework()
else:
    # Setup slurm launcher
    framework.launch_slurm()
