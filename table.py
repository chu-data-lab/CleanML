import json
import pandas as pd
import numpy as np
import utils
import re
from collections import Counter

def form_all(result_dict, key, precision=6):
    if key in result_dict.keys():
        res = result_dict[key]
        s = ",".join(["{1:.{0}f}".format(precision, r) for r in res])
    else:
        s = "nan"
    return s

def form_mean_sd(result_dict, key, precision=6):
    if key in result_dict.keys():
        res = result_dict[key]
        mean = "{1:.{0}f}".format(precision, np.mean(res))
        sd = "{1:.{0}f}".format(precision, np.std(res))
        s = "{}±{}".format(mean, sd)
    else:
        s = "nan"
    return s

def summarize(result, form, precision=6):
    comb = utils.combine(result)
    summary = {}
    for k, v in comb.items():
        dataset, error, file, model = k
    
        test_files = sorted(utils.get_test_files(error, file))[::-1]
        test_acc_keys = ['{}_test_acc'.format(f) for f in test_files if f[0] == 'c']
        test_f1_keys = ['{}_test_f1'.format(f) for f in test_files if f[0] == 'c']

        train_acc = form(v, 'train_acc', precision)
        val_acc = form(v, 'val_acc', precision)
        dirty_test_acc = form(v, 'dirty_test_acc', precision)
        dirty_test_f1 = form(v, 'dirty_test_f1', precision)
        clean_test_acc = "/".join([form(v, test_k) for test_k in test_acc_keys])
        clean_test_f1 = "/".join([form(v, test_k) for test_k in test_f1_keys])

        summary[k + ('train_acc',)] = train_acc
        summary[k + ('val_acc',)] = val_acc
        summary[k + ('dirty_test_acc',)] = dirty_test_acc
        summary[k + ('clean_test_acc',)] = clean_test_acc
        summary[k + ('dirty_test_f1',)] = dirty_test_f1
        summary[k + ('clean_test_f1',)] = clean_test_f1
    return summary

def form_mean(result_dict, key, precision=6):
    if key in result_dict.keys():
        res = result_dict[key]
        mean = "{1:.{0}f}".format(precision, np.mean(res))
        return mean
    else:
        return "nan"

if __name__ == '__main__':
    result = utils.load_result()
    # summary = summarize(result, form_all)
    summary = summarize(result, form_mean_sd)
    utils.dict_to_xls(summary, [0, 2], [3, 4], 1, './test.xls')