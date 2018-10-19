import  pandas as pd
import numpy as np
import os
import sys

"""
Deleting missing values: MVCleaner(method = 'delete')
Imuputing missing values: MVCleaner(method = 'impute', num = 'mean', cat = 'mode')

"""
class MVCleaner(object):
    def __init__(self, method='delete', **kwargs):
        self.method = method
        self.kwargs = kwargs
        if method == 'impute':
            if 'num' not in kwargs or 'cat' not in kwargs:
                print("Must give imputation method for numerical and categorical data")
                sys.exit(1)
            self.tag = "impute_{}_{}".format(kwargs['num'], kwargs['cat'])
        else:
            self.tag = "delete"

    def detect(self, df):
        return df.isnull()

    def repair(self, df, neighbors=3):
        if self.method == 'delete':
            df_clean = df.dropna()

        if self.method == 'impute':
            num_method = self.kwargs['num']
            cat_method = self.kwargs['cat']

            num_df = df.select_dtypes(include='number')
            cat_df = df.select_dtypes(exclude='number')
            if num_method == "mean":
                num_df_clean = num_df.fillna(num_df.mean())
            if num_method == "median":
                num_df_clean = num_df.fillna(num_df.median())
            if num_method == "mode":
                num_df_clean = num_df.apply(lambda x: x.fillna(x.value_counts().index[0]))
            if num_method == 'knn':
                import fancyimpute
                num_df_clean = pd.DataFrame(fancyimpute.KNN(neighbors).fit_transform(num_df))
                num_df_clean.columns = df.columns
            
            if cat_method == "mode":
                cat_df_clean = cat_df.apply(lambda x: x.fillna(x.value_counts().index[0]))
            if cat_method == "dummy":
                cat_df_clean = cat_df.fillna('missing')

            df_clean = pd.concat([num_df_clean, cat_df_clean], axis=1).reindex(columns = df.columns)
        return df_clean

    def clean(self, df, neighbors=3):
        mv_mat = self.detect(df)
        df_clean = self.repair(df, neighbors=neighbors)
        return df_clean, mv_mat