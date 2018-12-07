import pandas as pd

class DuplicatesCleaner(object):
    """docstring for DupCleaner"""
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
        