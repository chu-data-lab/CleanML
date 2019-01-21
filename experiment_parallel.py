from train import train_and_evaluate
from preprocess import preprocess
import numpy as np
import utils
import json
import argparse
import config
from multiprocessing import Pool

def one_experiment(dataset, error_type, train_file, model, seed, n_jobs=1):
    """ One experiment on the datase given an error type, a train file, a model and a seed
        
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

def one_experiment_helper(args):
    dataset, error, train_file, model, seed, in_cpu, key = args
    print("Processing {}".format(key))
    return one_experiment(dataset, error, train_file, model, seed, in_cpu)

def experiment_parallel(dataset, n_retrain=5, seed=1, n_jobs=1, nosave=True):
    """ Run all experiments on one dataset.

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
    result = utils.load_result()
    
    # run experiments
    args = []
    keys = []
    for error in dataset["error_types"]:
        for train_file in utils.get_train_files(error):
            for model in config.models:
                for seed in seeds:
                    version = utils.get_version(utils.get_dir(dataset, error, train_file))
                    key = "/".join((dataset['data_dir'], 'v'+str(version), error, train_file, model['name'], str(seed)))
                    if key not in result.keys():
                        args.append((dataset, error, train_file, model, seed, 1, key)) 
                        keys.append(key)

    if len(args) > 0:
        pool = Pool(n_jobs)
        res = pool.map(one_experiment_helper, args)
        
        if not nosave:
            utils.save_result_list(keys, res)