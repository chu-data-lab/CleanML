"""Main function"""

import numpy as np
import utils
import json
import argparse
import datetime
import time
import config
from experiment import experiment
from relation import populate

parser = argparse.ArgumentParser()
parser.add_argument('--run_experiments', default=False, action='store_true')
parser.add_argument('--run_analysis', default=False, action='store_true')
parser.add_argument('--cpu', default=1, type=int)
parser.add_argument('--log', default=False, action='store_true')
parser.add_argument('--dataset', default=None)
parser.add_argument('--nosave', default=False, action='store_true')
parser.add_argument('--alpha', default=0.05, type=float)

args = parser.parse_args()

# run experiments on datasets
if args.run_experiments:
    datasets = [utils.get_dataset(args.dataset)] if args.dataset is not None else config.datasets
    experiment(datasets, args.log, args.cpu, args.nosave)

# run analysis on results
if args.run_analysis:
    populate([args.alpha])  