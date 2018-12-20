import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import sys

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

    def fit(self, df):
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

    def clean(self, df):
        if not self.is_fit:
            print('Must fit before clean.')
            sys.exit()
        mv_mat = self.detect(df)
        df_clean = self.repair(df)
        return df_clean, mv_mat

class DuplicatesCleaner(object):
    def __init__(self):
        super(DuplicatesCleaner, self).__init__()
    
    def detect(self, df, keys):
        key_col = pd.DataFrame(df, columns=keys)
        is_dup = key_col.duplicated(keep='first')
        is_dup = pd.DataFrame(is_dup, columns=['is_dup'])
        return is_dup

    def repair(self, df, is_dup):
        not_dup = is_dup == False
        df_clean = df[not_dup]
        return df_clean

    def clean(self, df, keys):
        is_dup = self.detect(df, keys)
        df_clean = self.repair(df, is_dup)
        return df_clean, is_dup

class InconsistencyCleaner(object):
    def __init__(self):
        super(InconsistencyCleaner, self).__init__()

    def fit(self, dirty_train, clean_train):
        N, m = dirty_train.shape
        dirty_train = dirty_train.values
        clean_train = clean_train.values
        mask = (dirty_train != clean_train)
        dirty = dirty_train[mask]
        clean = clean_train[mask]
        self.incon_dict = dict(zip(dirty, clean))

    def clean(self, df):
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

def iso_forest(x, contamination=0.01):
    # Isolation Forest (Univariate)
    iso_forest = IsolationForest(contamination=contamination)
    iso_forest.fit(x.reshape(-1, 1))
    return lambda y: (iso_forest.predict(y.reshape(-1, 1)) == -1)

def LOF(x, contamination=0.1):
    # Local Outlier Factor (Multivariate)
    lof = LocalOutlierFactor(contamination=contamination)
    lof.fit(x)
    return lambda y: (lof.predict(y) == -1)

def DBscan(X, eps=3.0, min_samples=10): #(Deprecated)
    # DBscan (Multivariate)
    X_scale = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scale)
    is_outlier = (db.labels_ == -1)
    return is_outlier

class OutlierCleaner(object):
    def __init__(self, detect, repairer=MVCleaner('delete'), **kwargs):
        super(OutlierCleaner, self).__init__()
        self.detect_method = detect
        self.univariate = {'SD': SD, 'IQR':IQR, 'iso_forest':iso_forest}
        self.multivariate = {'LOF':LOF, 'DBscan':DBscan}
        self.is_uni = (detect in self.univariate.keys())
        self.repairer = repairer
        self.kwargs = kwargs
        self.tag = "{}_{}".format(detect, repairer.tag)
        self.is_fit = False

        if not self.is_uni and repairer.method != 'delete':
            print('Must repair outliers by deleting for multivariate detection approaches')
            sys.exit(1)
    
    def fit(self, df):
        num_df = df.select_dtypes(include='number')
        cat_df = df.select_dtypes(exclude='number')

        X = num_df.values
        m = X.shape[1]
        
        if self.is_uni:
            method = self.univariate[self.detect_method]
            self.detectors = []
            
            for i in range(m):
                x = X[:, i]
                detector = method(x, **self.kwargs)
                self.detectors.append(detector)
        else:
            method = self.multivariate[self.detect_method]
            self.detectors = method(X, **kwargs)

        ind = self.detect(df)
        df_copy = df.copy()
        if self.is_uni:
            df_copy[ind] = np.nan
            self.repairer.fit(df_copy)
        self.is_fit = True

    def detect(self, df, verbose=False, show=False):
        num_df = df.select_dtypes(include='number')
        cat_df = df.select_dtypes(exclude='number')
        X = num_df.values
        m = X.shape[1]

        if self.is_uni:
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
        else:
            ind = self.detectors(X)

        if verbose:
            if self.is_uni:
                print("Outlier percentage:", np.mean(ind.any(axis=1)))
            else:
                print("Outlier percentage:", np.mean(ind))

        if show and self.is_uni:
            plt.figure()
            self.plot_outliers(df, ind)
            plt.title(self.detect_method)
            plt.xlabel('Features')
            plt.ylabel('Normalized values')
        return ind

    def repair(self, df, ind):
        df_copy = df.copy()
        if self.is_uni:
            df_copy[ind] = np.nan
            df_clean, _ = self.repairer.clean(df_copy)
        else:
            clean_mat = (ind == 0)
            df_clean = df_copy[clean_mat]
        return df_clean

    def clean(self, df, verbose=False, show=False, ignore=None):
        if not self.is_fit:
            print("Must fit before clean")
            sys.exit()
        ind = self.detect(df, verbose, show)
        if ignore is not None:
            ind.loc[:, ignore] = False
        df_clean = self.repair(df, ind)
        return df_clean, ind

    def plot_outliers(self, df, ind):
        num_df = df.select_dtypes(include='number')
        X_scale = StandardScaler().fit_transform(num_df)
        N, m = X_scale.shape

        ind = ind[num_df.columns].values
        clean_mat = (ind == 0)

        for i in range(m):
            is_outlier = ind[:, i].reshape(-1,1)
            is_clean = clean_mat[:, i].reshape(-1,1)
            x = X_scale[:, i].reshape(-1,1)
            x_clean = x[is_clean]
            x_outlier = x[is_outlier]
            plt.scatter(np.ones_like(x_clean)*i, x_clean , c='blue')
            plt.scatter(np.ones_like(x_outlier)*i, x_outlier, c='red')
