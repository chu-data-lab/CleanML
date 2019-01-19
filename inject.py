import pandas as pd
import config
import os
import argparse
import utils

def uniform_class_noise(df, label, percentage=0.05, random_state=123):
    """Uniform class noise in a binary classification dataset. 
    x% of the examples are corrupted. 
    The class labels of these examples are randomly replaced by another one from the M classes.
    - flip 5% in each class, in total 5% of the labels are changed

    Args:
        df: pandas dataframe
        percentage: the percentage to corrupt, percentage = X
        label: the column of label
    """
    ## load in csv
    dist = df[label].value_counts(ascending=True)
    # print('class distribution before injection:\n', dist)
    
    classes = list(dist.index)

    ## label == 1
    train1 = df[df[label]==classes[1]].copy()
    train1.loc[train1.sample(frac=percentage, random_state=random_state).index, label] = classes[0]
    
    ## label == 0
    train0 = df[df[label]==classes[0]].copy()
    train0.loc[train0.sample(frac=percentage, random_state=random_state).index, label] = classes[1]
    
    ## append the noisy sets
    uniform_df = train1.append(train0)
    # print('\nclass distribution after uniform injection:\n', uniform_df[label].value_counts(ascending=True))
    return uniform_df

def pairwise_class_noise(df, label, percentage=0.05, random_state=123):
    """ Pairwise class noise. 
    Let X be the majority class and Y the second
    majority class, an example with the label X has a probability of x/100 of
    being incorrectly labeled as Y .
    - flip 5% of the labels in class A and keep the labels for class B
    - flip 5% of the labels in class B and keep the labels for class A
    
    Args:
        df: pandas dataframe
        percentage: the percentage to corrupt, percentage = X
        label, the column of label
        class_to_flip, the class label to corrupt
    """
    ## load in csv
    dist = df[label].value_counts(ascending=True)
    # print('class distribution before injection:\n', dist)

    classes = list(dist.index)

    flip_major = df.copy()
    flip_major.loc[df[df[label]==classes[1]].sample(frac=percentage, random_state=random_state).index, label] = classes[0]
    flip_minor = df.copy()
    flip_minor.loc[df[df[label]==classes[0]].sample(frac=percentage, random_state=random_state).index, label] = classes[1]

    # print('\nclass distribution after injection (flip majority class):\n', flip_major[label].value_counts(ascending=True))
    # print('\nclass distribution after injection (flip minority class):\n', flip_minor[label].value_counts(ascending=True))
    return flip_major, flip_minor

def inject(dataset):
    """ Inject mislabels
        Args:
            dataset (dict): dataset dict in config
    """
        # create saving folder
    save_dir = utils.get_dir(dataset, 'mislabel', create_folder=True)

    # load clean data
    if 'missing_values' in dataset['error_types']:
        # if raw dataset has missing values, use dataset with mv deleted in missing value folder 
        clean_path_pfx = utils.get_dir(dataset, 'missing_values', 'delete')
    else:
        clean_path_pfx = utils.get_dir(dataset, 'raw', 'dirty')
    clean_train, clean_test, version = utils.load_dfs(dataset, clean_path_pfx, return_version=True)
    
    # save clean
    clean_path_pfx = os.path.join(save_dir, 'clean')
    utils.save_dfs(clean_train, clean_test, clean_path_pfx, version)


    label = dataset['label']
    # uniform flip
    uniform_train = uniform_class_noise(clean_train, label)
    uniform_test = uniform_class_noise(clean_test, label)

    # pairwise flip
    major_train, minor_train = pairwise_class_noise(clean_train, label)
    major_test, minor_test = pairwise_class_noise(clean_test, label)

    dirty_path_pfx = os.path.join(save_dir, 'dirty_uniform')
    utils.save_dfs(uniform_train, uniform_test, dirty_path_pfx, version)
    dirty_path_pfx = os.path.join(save_dir, 'dirty_major')
    utils.save_dfs(major_train, major_test, dirty_path_pfx, version)
    dirty_path_pfx = os.path.join(save_dir, 'dirty_minor')
    utils.save_dfs(minor_train, minor_test, dirty_path_pfx, version)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()

    # datasets to be inject, inject all datasets with error type mislabel if not specified
    datasets = [utils.get_dataset(args.dataset)] if args.dataset is not None else config.datasets

    # clean datasets
    for dataset in datasets:
        if 'mislabel' in dataset['error_types']:
            inject(dataset)