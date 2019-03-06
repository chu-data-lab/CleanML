"""Run experiments"""
from train import train_and_evaluate
from preprocess import preprocess
import numpy as np
import utils
import json
import argparse
import config
import datetime
from init import init
from clean import clean
import time
import logging

def one_search_experiment(dataset, error_type, train_file, model, seed, n_jobs=1):
    """One experiment on the datase given an error type, a train file, a model and a random search seed
        
    Args:
        dataset (dict): dataset dict in config.py
        error_type (string): error type
        train_file (string): filename of training set (dirty or clean) 
        model (dict): ml model dict in model.py
        seed (int): seed for this experiment
    """
    np.random.seed(seed)
    # generate random seeds for down sample and training
    down_sample_seed, train_seed = np.random.randint(1000, size=2)

    # load and preprocess data
    X_train, y_train, X_test_list, y_test_list, test_files = \
        preprocess(dataset, error_type, train_file, normalize=True, down_sample_seed=down_sample_seed)

    # train and evaluate
    result = train_and_evaluate(X_train, y_train, X_test_list, y_test_list, test_files, model, n_jobs=n_jobs, seed=train_seed)
    return result

def one_split_experiment(dataset, n_retrain=5, seed=1, n_jobs=1, nosave=True):
    """Run experiments on one dataset for one split.

    Args:
        dataset (dict): dataset dict in config.py
        models (list): list of model dict in model.py
        nosave (bool): whether not save results
        seed: experiment seed
        n_retrain: times of repeated experiments
    """
    # generate seeds for n experiments
    np.random.seed(seed)
    seeds = np.random.randint(10000, size=n_retrain)

    # load result dict
    result = utils.load_result(dataset['data_dir'])

    # run experiments
    for error in dataset["error_types"]:
        for train_file in utils.get_train_files(error):
            for model in config.models:
                for seed in seeds:
                    version = utils.get_version(utils.get_dir(dataset, error, train_file))
                    key = "/".join((dataset['data_dir'], 'v'+str(version), error, train_file, model['name'], str(seed)))

                    if key in result.keys():
                        print("Ignore experiment {} that has been completed before.".format(key))
                        continue
        
                    print("{} Processing {}".format(datetime.datetime.now(), key)) 
                    res = one_search_experiment(dataset, error, train_file, model, seed, n_jobs=n_jobs)
                    if not nosave:
                        utils.save_result(dataset['data_dir'], key, res)

def experiment(datasets, log=False, n_jobs=1, nosave=False):
    """Run expriments on all datasets for all splits"""
    # set logger for experiments
    if log:
        logging.captureWarnings(True)
        logging.basicConfig(filename='logging_{}.log'.format(datetime.datetime.now()), level=logging.DEBUG)

    # set seeds for experiments
    np.random.seed(config.root_seed)
    split_seeds = np.random.randint(10000, size=config.n_resplit)
    experiment_seed = np.random.randint(10000)

    # run experiments
    for dataset in datasets:
        if log:
            logging.debug("{}: Experiment on {}".format(datetime.datetime.now(), dataset['data_dir']))

        for i, seed in enumerate(split_seeds):
            if utils.check_completed(dataset, seed, experiment_seed):
                print("Ignore {}-th experiment on {} that has been completed before.".format(i, dataset['data_dir']))
                continue
            tic = time.time()
            init(dataset, seed=seed, max_size=config.max_size)
            clean(dataset)
            one_split_experiment(dataset, n_retrain=config.n_retrain, n_jobs=n_jobs, nosave=nosave, seed=experiment_seed)
            toc = time.time()
            t = (toc - tic) / 60
            remaining = t*(len(split_seeds)-i-1) 
            if log:
                logging.debug("{}: {}-th experiment takes {} min. Estimated remaining time: {} min".format(datetime.datetime.now(), i, t, remaining))