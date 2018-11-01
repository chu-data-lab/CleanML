import pandas as pd
import numpy as np

class InconsistencyCleaner(object):
    """docstring for InconsistencyCleaner"""
    def __init__(self):
        super(InconsistencyCleaner, self).__init__()

    def fit(self, dirty_train, clean_train):
        N, m = dirty_train.shape
        dirty_train = dirty_train.values
        clean_train = clean_train.values
        self.incon_dict = {}
        mask = dirty_train != clean_train
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



