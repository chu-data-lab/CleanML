from init import init
from experiment import experiment
from clean import clean
import numpy as np
import utils
import json
import argparse
import logging
import datetime
import time
import config

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', default=1, type=int)
parser.add_argument('--log', default=False, action='store_true')
parser.add_argument('--nosave', default=False, action='store_true')
parser.add_argument('--dataset', default=None)

args = parser.parse_args()

if args.log:
    logging.captureWarnings(True)
    logging.basicConfig(filename='logging_{}.log'.format(datetime.datetime.now()), level=logging.DEBUG)

if __name__ == '__main__':
    np.random.seed(config.root_seed)
    split_seeds = np.random.randint(10000, size=config.n_resplit)
    experiment_seed = np.random.randint(10000)
    datasets = [utils.get_dataset(args.dataset)] if args.dataset is not None else config.datasets
    
    for dataset in datasets:
        logging.debug("{}: Experiment on {}".format(datetime.datetime.now(), dataset['data_dir']))

        for i, seed in enumerate(split_seeds):
            if utils.check_completed(dataset, seed, experiment_seed):
                print("Ignore {}-th experiment on {} that has been completed before.".format(i, dataset['data_dir']))
                continue
            tic = time.time()
            init(dataset, seed=seed, max_size=config.max_size)
            clean(dataset)
            experiment(dataset, n_retrain=config.n_retrain, n_jobs=args.cpu, nosave=args.nosave, seed=experiment_seed)
            toc = time.time()
            t = (toc - tic) / 60
            remaining = t*(len(split_seeds)-i-1) 
            logging.debug("{}: {}-th experiment takes {} min. Estimated remaining time: {} min".format(datetime.datetime.now(), i, t, remaining))