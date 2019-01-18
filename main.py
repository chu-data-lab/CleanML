from init import init
from experiment import experiment
import numpy as np
import utils
import json
import argparse
import logging
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--log', default=False, action='store_true')
parser.add_argument('--nosave', default=False, action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    dataset_names = ["Titanic"]
    init(dataset, seed=split_seed)
    clean(dataset)
    experiment(dataset, jobs=args.cpu, seed=experiment_seed, nosave=args.nosave)
    












# print log info
logging.info("Processing {}".format(key))




## setup


# dataset_names = ['KDD']
# model_names = ["guassian_naive_bayes"]

## save logging info
if args.log:
    logging.captureWarnings(True)
    logging.basicConfig(filename='logging_{}.log'.format(datetime.datetime.now()),level=logging.DEBUG)

"""
            result (dict): dict to save experimental results
                key (string): dataset_name/error_type/train_file/model_name/seed
                value (dict): metric_name: metric
"""

    # ignore if experiment has been ran before 
