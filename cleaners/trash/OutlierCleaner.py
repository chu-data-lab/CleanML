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

def SD(x, nstd=3.0):
    # Standard Deviaiton Method (Univariate)
    mean, std = np.mean(x), np.std(x)
    cut_off = std * nstd
    lower, upper = mean - cut_off, mean + cut_off
    is_outlier = (x > upper) | (x < lower)
    return is_outlier

def IQR(x, k=1.5):
    # Interquartile Range (Univariate)
    q25, q75 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = q75 - q25
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    is_outlier = (x > upper) | (x < lower)
    return is_outlier

def iso_forest(x, contamination=0.01):
    # Isolation Forest (Univariate)
    iso_forest = IsolationForest(contamination=contamination)
    iso_forest.fit(x.reshape(-1, 1))
    is_outlier = (iso_forest.predict(x.reshape(-1, 1)) == -1)
    return is_outlier

def LOF(x, contamination=0.1):
    # Local Outlier Factor (Multivariate)
    lof = LocalOutlierFactor(contamination=contamination)
    is_outlier = (lof.fit_predict(x) == -1)
    return is_outlier

def DBscan(X, eps=3.0, min_samples=10):
    # DBscan (Multivariate)
    X_scale = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scale)
    is_outlier = (db.labels_ == -1)
    return is_outlier

class OutlierCleaner(object):
    def __init__(self, detect = 'iso_forest', repair = 'delete', **kwargs):
        super(OutlierCleaner, self).__init__()
        self.detect_method = detect
        self.repair_method = repair
        self.univariate = {'SD': SD, 'IQR':IQR, 'iso_forest':iso_forest}
        self.multivariate = {'LOF':LOF, 'DBscan':DBscan}
        self.is_uni = (detect in self.univariate.keys())
        self.kwargs = kwargs

        if not self.is_uni and repair != 'delete':
            print('Must repair outliers by deleting for multivariate detection approaches')
            sys.exit(1)

    def detect(self, df):
        method = self.detect_method
        kwargs = self.kwargs

        num_df = df.select_dtypes(include='number')
        cat_df = df.select_dtypes(exclude='number')
        outlier_mat_num = np.zeros_like(num_df).astype('bool')
        outlier_mat_cat = np.zeros_like(cat_df).astype('bool')

        N, m = num_df.shape

        if self.is_uni:
            detector = self.univariate[method]
            for i in range(m):
                x = num_df.iloc[:, i]
                if len(kwargs) == 0:
                    is_outlier = detector(x)
                else:
                    is_outlier = detector(x, **kwargs)
                outlier_mat_num[:, i] = is_outlier
        else:
            detector = self.multivariate[method]
            if len(kwargs) == 0:
                outlier_mat = detector(X)
            else:
                outlier_mat_num = detector(X, **kwargs)

        outlier_mat_num = pd.DataFrame(outlier_mat_num, columns=num_df.columns)
        outlier_mat_cat = pd.DataFrame(outlier_mat_cat, columns=cat_df.columns)
        outlier_mat = pd.concat([outlier_mat_num, outlier_mat_cat], axis=1).reindex(columns=df.columns)

        return outlier_mat

    def repair(self, df, outlier_mat):
        method = self.repair_method
        df_copy = df.copy()

        if self.is_uni:
            df_copy[outlier_mat] = np.nan
            mv_cleaner = MVCleaner(method=self.repair_method, **self.kwargs)
            df_clean, _ = mv_cleaner.clean(df_copy)
        else:
            clean_mat = (outlier_mat == 0)
            df_clean = df_copy[clean_mat]

        return df_clean

    def clean(self, df, verbose=False, show=False):
        outlier_mat = self.detect(df)
        df_clean = self.repair(df, outlier_mat)

        if verbose:
            if self.is_uni:
                print("Outlier percentage:", np.mean(outlier_mat.any(axis=1)))
            else:
                print("Outlier percentage:", np.mean(outlier_mat))

        if show and self.is_uni:
            plt.figure()
            self.plot_outliers(X, outlier_mat)
            plt.title(self.detect_method)
            plt.xlabel('Features')
            plt.ylabel('Normalized values')
            
        return df_clean, outlier_mat

    def plot_outliers(self, X, outlier_mat):
        X_scale = StandardScaler().fit_transform(X)
        N, m = X_scale.shape
        clean_mat = (outlier_mat == 0)

        for i in range(m):
            is_outlier = outlier_mat[:, i].reshape(-1,1)
            is_clean = clean_mat[:, i].reshape(-1,1)
            x = X_scale[:, i].reshape(-1,1)
            x_clean = x[is_clean]
            x_outlier = x[is_outlier]
            plt.scatter(np.ones_like(x_clean)*i, x_clean , c='blue')
            plt.scatter(np.ones_like(x_outlier)*i, x_outlier, c='red')
