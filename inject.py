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
    major_save_dir = utils.makedirs([config.data_dir, dataset["data_dir"] + "_major", 'raw'])
    minor_save_dir = utils.makedirs([config.data_dir, dataset["data_dir"] + "_minor", 'raw'])
    uniform_save_dir = utils.makedirs([config.data_dir, dataset["data_dir"] + "_uniform", 'raw'])

    # load clean data
    clean_path = utils.get_dir(dataset, 'raw', 'raw.csv')
    clean = utils.load_df(dataset, clean_path)
    clean = clean.dropna().reset_index(drop=True)

    major_clean_path = os.path.join(major_save_dir, 'mislabel_clean_raw.csv')
    minor_clean_path = os.path.join(minor_save_dir, 'mislabel_clean_raw.csv')
    uniform_clean_path = os.path.join(uniform_save_dir, 'mislabel_clean_raw.csv')
    clean.to_csv(major_clean_path, index=False)
    clean.to_csv(minor_clean_path, index=False)
    clean.to_csv(uniform_clean_path, index=False)

    label = dataset['label']

    # uniform flip
    uniform = uniform_class_noise(clean, label)
    # pairwise flip
    major, minor = pairwise_class_noise(clean, label)

    major_raw_path = os.path.join(major_save_dir, 'raw.csv')
    minor_raw_path = os.path.join(minor_save_dir, 'raw.csv')
    uniform_raw_path = os.path.join(uniform_save_dir, 'raw.csv')
    
    major.to_csv(major_raw_path, index=False)
    minor.to_csv(minor_raw_path, index=False)
    uniform.to_csv(uniform_raw_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()

    # datasets to be inject, inject all datasets with error type mislabel if not specified
    datasets = [utils.get_dataset(args.dataset)] 
    
    # clean datasets
    for dataset in datasets:
        inject(dataset)