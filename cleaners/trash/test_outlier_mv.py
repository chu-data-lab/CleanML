from sklearn.datasets import load_breast_cancer
from OutlierCleaner import OutlierCleaner
from MVCleaner import MVCleaner
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

bc_data = load_breast_cancer()
X, y = bc_data.data, bc_data.target

df = pd.DataFrame(X)
out_repairer = MVCleaner(method='impute', num='mean', cat='mode')
out_cleaner = OutlierCleaner(detect='SD', repairer=out_repairer)
df_clean, df_outlier_mat = out_cleaner.clean(df, verbose = True)


# out_cleaner = OutlierCleaner(detect = 'IQR', repair = 'mean', k = 3)
# X_clean, outlier_mat = out_cleaner.clean(X, verbose = True, show = True)
# out_cleaner = OutlierCleaner(detect = 'iso_forest', repair = 'median')
# X_clean, outlier_mat = out_cleaner.clean(X, verbose = True, show = True)
# out_cleaner = OutlierCleaner(detect = 'LOF', repair = 'delete')
# X_clean, outlier_mat = out_cleaner.clean(X, verbose = True)
# out_cleaner = OutlierCleaner(detect = 'DBscan', repair = 'delete', min_samples = 3)
# X_clean, outlier_mat = out_cleaner.clean(X, verbose = True)
# plt.show()