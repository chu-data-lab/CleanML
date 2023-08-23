# define the domain of error type 
from .clean_method import *

# details of each error type
missing_values = {
    "name": "missing_values",
    "clean_methods": {"delete": MVCleaner("delete"),
                      "impute_holoclean": MVHoloCleaner(),
                      "impute_mean_mode": MVCleaner("impute", num="mean", cat="mode"),
                      "impute_mean_dummy": MVCleaner("impute", num="mean", cat="dummy"),
                      "impute_median_mode": MVCleaner("impute", num="median", cat="mode"),
                      "impute_median_dummy": MVCleaner("impute", num="median", cat="dummy"),
                      "impute_mode_mode": MVCleaner("impute", num="mode", cat="mode"),
                      "impute_mode_dummy": MVCleaner("impute", num="mode", cat="dummy"),
                    }
}

outliers = {
    "name": "outliers",
    "clean_methods": {"clean_HC_impute_holoclean": OutlierHoloCleaner(),
                      "clean_SD_impute_mean_dummy": OutlierCleaner(detect_method="SD", repairer=MVCleaner("impute", num="mean", cat="dummy")),
                      "clean_SD_impute_mode_dummy": OutlierCleaner(detect_method="SD", repairer=MVCleaner("impute", num="mode", cat="dummy")),
                      "clean_SD_impute_median_dummy": OutlierCleaner(detect_method="SD", repairer=MVCleaner("impute", num="median", cat="dummy")),
                      "clean_IQR_impute_mean_dummy": OutlierCleaner(detect_method="IQR", repairer=MVCleaner("impute", num="mean", cat="dummy")),
                      "clean_IQR_impute_mode_dummy": OutlierCleaner(detect_method="IQR", repairer=MVCleaner("impute", num="mode", cat="dummy")),
                      "clean_IQR_impute_median_dummy": OutlierCleaner(detect_method="IQR", repairer=MVCleaner("impute", num="median", cat="dummy")),
                      "clean_IF_impute_mean_dummy": OutlierCleaner(detect_method="IF", repairer=MVCleaner("impute", num="mean", cat="dummy")),
                      "clean_IF_impute_mode_dummy": OutlierCleaner(detect_method="IF", repairer=MVCleaner("impute", num="mode", cat="dummy")),
                      "clean_IF_impute_median_dummy": OutlierCleaner(detect_method="IF", repairer=MVCleaner("impute", num="median", cat="dummy")),
                    }
}

mislabel = {
    "name": "mislabel",
    "clean_methods": {"cleanlab": MislabelCleaner()}
}

duplicates = {
    "name": "duplicates",
    "clean_methods": {"clean": DuplicatesCleaner(), "AutoER": AutoERCleaner()}
}

inconsistency = {
    "name": "inconsistency",
    "clean_methods": {"clean": InconsistencyCleaner()}
}

# domain of error types
error_types = [missing_values, outliers, mislabel, inconsistency, duplicates]