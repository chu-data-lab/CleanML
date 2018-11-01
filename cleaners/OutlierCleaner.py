import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from .MVCleaner import MVCleaner
import sys

brk = False
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
