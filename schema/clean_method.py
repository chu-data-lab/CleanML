# define the domain of cleaning method
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import sys
import utils
import os

class MVCleaner(object):
    def __init__(self, method='delete', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.is_fit = False
        if method == 'impute':
            if 'num' not in kwargs or 'cat' not in kwargs:
                print("Must give imputation method for numerical and categorical data")
                sys.exit(1)
            self.tag = "impute_{}_{}".format(kwargs['num'], kwargs['cat'])
        else:
            self.tag = "delete"

    def detect(self, df):
        return df.isnull()

    def fit(self, dataset, df):
        if self.method == 'impute':
            num_method = self.kwargs['num']
            cat_method = self.kwargs['cat']
            num_df = df.select_dtypes(include='number')
            cat_df = df.select_dtypes(exclude='number')
            if num_method == "mean":
                num_imp = num_df.mean()
            if num_method == "median":
                num_imp = num_df.median()
            if num_method == "mode":
                num_imp = num_df.mode().iloc[0]

            if cat_method == "mode":
                cat_imp = cat_df.mode().iloc[0]
            if cat_method == "dummy":
                cat_imp = ['missing'] * len(cat_df.columns)
                cat_imp = pd.Series(cat_imp, index=cat_df.columns)
            self.impute = pd.concat([num_imp, cat_imp], axis=0)
        self.is_fit = True

    def repair(self, df):
        if self.method == 'delete':
            df_clean = df.dropna()

        if self.method == 'impute':
            df_clean = df.fillna(value=self.impute)
        return df_clean

    def clean_df(self, df):
        if not self.is_fit:
            print('Must fit before clean.')
            sys.exit()
        mv_mat = self.detect(df)
        df_clean = self.repair(df)
        return df_clean, mv_mat

    def clean(self, dirty_train, dirty_test):
        clean_train, indicator_train = self.clean_df(dirty_train)
        clean_test, indicator_test = self.clean_df(dirty_test)
        return clean_train, indicator_train, clean_test, indicator_test

class DuplicatesCleaner(object):
    def __init__(self):
        super(DuplicatesCleaner, self).__init__()

    def fit(self, dataset, df):
        self.keys = dataset['key_columns']
    
    def detect(self, df, keys):
        key_col = pd.DataFrame(df, columns=keys)
        is_dup = key_col.duplicated(keep='first')
        is_dup = pd.DataFrame(is_dup, columns=['is_dup'])
        return is_dup

    def repair(self, df, is_dup):
        not_dup = (is_dup.values == False)
        df_clean = df[not_dup]
        return df_clean

    def clean_df(self, df):
        is_dup = self.detect(df, self.keys)
        df_clean = self.repair(df, is_dup)
        return df_clean, is_dup

    def clean(self, dirty_train, dirty_test):
        clean_train, indicator_train = self.clean_df(dirty_train)
        clean_test, indicator_test = self.clean_df(dirty_test)
        return clean_train, indicator_train, clean_test, indicator_test

class InconsistencyCleaner(object):
    def __init__(self):
        super(InconsistencyCleaner, self).__init__()

    def fit(self, dataset, dirty_train):
        dirty_raw_path = utils.get_dir(dataset, 'raw', 'raw.csv')
        clean_raw_path = utils.get_dir(dataset, 'raw', 'inconsistency_clean_raw.csv')
        if not os.path.exists(clean_raw_path):
            print("Must provide clean version of raw data for cleaning inconsistency")
            sys.exit(1)
        dirty_raw = utils.load_df(dataset, dirty_raw_path)
        clean_raw = utils.load_df(dataset, clean_raw_path)
        N, m = dirty_raw.shape
        dirty_raw = dirty_raw.values
        clean_raw = clean_raw.values
        mask = (dirty_raw != clean_raw)
        dirty = dirty_raw[mask]
        clean = clean_raw[mask]
        self.incon_dict = dict(zip(dirty, clean))

    def clean_df(self, df):
        df_clean = df.copy()
        N, m = df_clean.shape
        indicator = np.zeros_like(df_clean).astype(bool)

        for i in range(N):
            for j in range(m):
                if df_clean.iloc[i, j] in self.incon_dict.keys():
                    df_clean.iloc[i, j] = self.incon_dict[df_clean.iloc[i, j]]
                    indicator[i, j] = True
        indicator = pd.DataFrame(indicator, columns=df.columns)
        return df_clean, indicator

    def clean(self, dirty_train, dirty_test):
        clean_train, indicator_train = self.clean_df(dirty_train)
        clean_test, indicator_test = self.clean_df(dirty_test)
        return clean_train, indicator_train, clean_test, indicator_test

def SD(x, nstd=3.0):
    # Standard Deviaiton Method (Univariate)
    mean, std = np.mean(x), np.std(x)
    cut_off = std * nstd
    lower, upper = mean - cut_off, mean + cut_off
    return lambda y: (y > upper) | (y < lower)

def IQR(x, k=1.5):
    # Interquartile Range (Univariate)
    q25, q75 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = q75 - q25
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    return lambda y: (y > upper) | (y < lower)

def IF(x, contamination=0.01):
    # Isolation Forest (Univariate)
    IF = IsolationForest(contamination=contamination, behaviour='new')
    IF.fit(x.reshape(-1, 1))
    return lambda y: (IF.predict(y.reshape(-1, 1)) == -1)

class OutlierCleaner(object):
    def __init__(self, detect_method, repairer=MVCleaner('delete'), **kwargs):
        super(OutlierCleaner, self).__init__()
        detect_fn_dict = {'SD':SD, 'IQR':IQR, "IF":IF}
        self.detect_method = detect_method
        self.detect_fn = detect_fn_dict[detect_method]
        self.repairer = repairer
        self.kwargs = kwargs
        self.tag = "{}_{}".format(detect_method, repairer.tag)
        self.is_fit = False
    
    def fit(self, dataset, df):
        num_df = df.select_dtypes(include='number')
        cat_df = df.select_dtypes(exclude='number')
        X = num_df.values
        m = X.shape[1]

        self.detectors = []
        for i in range(m):
            x = X[:, i]
            detector = self.detect_fn(x, **self.kwargs)
            self.detectors.append(detector)

        ind = self.detect(df)
        df_copy = df.copy()
        df_copy[ind] = np.nan
        self.repairer.fit(dataset, df_copy)
        self.is_fit = True

    def detect(self, df):
        num_df = df.select_dtypes(include='number')
        cat_df = df.select_dtypes(exclude='number')
        X = num_df.values
        m = X.shape[1]

        ind_num = np.zeros_like(num_df).astype('bool')
        ind_cat = np.zeros_like(cat_df).astype('bool')
        for i in range(m):
            x = X[:, i]
            detector = self.detectors[i]
            is_outlier = detector(x)
            ind_num[:, i] = is_outlier

        ind_num = pd.DataFrame(ind_num, columns=num_df.columns)
        ind_cat = pd.DataFrame(ind_cat, columns=cat_df.columns)
        ind = pd.concat([ind_num, ind_cat], axis=1).reindex(columns=df.columns)
        return ind

    def repair(self, df, ind):
        df_copy = df.copy()
        df_copy[ind] = np.nan
        df_clean, _ = self.repairer.clean_df(df_copy)
        return df_clean

    def clean_df(self, df, ignore=None):
        if not self.is_fit:
            print("Must fit before clean")
            sys.exit()
        ind = self.detect(df)
        if ignore is not None:
            ind.loc[:, ignore] = False
        df_clean = self.repair(df, ind)
        return df_clean, ind

    def clean(self, dirty_train, dirty_test):
        clean_train, indicator_train = self.clean_df(dirty_train)
        clean_test, indicator_test = self.clean_df(dirty_test)
        return clean_train, indicator_train, clean_test, indicator_test

class MislabelCleaner(object):
    def __init__(self):
        super(MislabelCleaner, self).__init__()

    def fit(self, dataset, dirty_train):
        index_train_path = utils.get_dir(dataset, 'raw', 'idx_train.csv')
        index_test_path = utils.get_dir(dataset, 'raw', 'idx_test.csv')
        index_train = pd.read_csv(index_train_path).values.reshape(-1)
        index_test = pd.read_csv(index_test_path).values.reshape(-1)
        clean_path = utils.get_dir(dataset, 'raw', 'mislabel_clean_raw.csv')
        clean = utils.load_df(dataset, clean_path)
        self.clean_train = clean.loc[index_train, :]
        self.clean_test = clean.loc[index_test, :]

    def clean(self, dirty_train, dirty_test):
        indicator_train = pd.DataFrame(np.any(self.clean_train.values != dirty_train.values, axis=1), columns=['indicator'])
        indicator_test = pd.DataFrame(np.any(self.clean_test.values != dirty_test.values, axis=1), columns=['indicator'])
        return self.clean_train, indicator_train, self.clean_test, indicator_test