import  pandas as pd
import numpy as np
import os
import sys

"""
Deleting missing values: MVCleaner(method='delete')
Imuputing missing values: MVCleaner(method='impute', num='mean', cat='mode')

"""
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