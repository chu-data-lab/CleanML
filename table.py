import json
import pandas as pd
import numpy as np
import utils
import re
from collections import Counter

class TableGenerator(object):
    def __init__(self, results):
        # self.results = results
        # self.dicts={tuple(k.split('/')): v.split('/') for k, v in self.results.items()}
        self.dicts = results
        self.getErrortype()

    def getErrortype(self):
        keys=self.dicts.keys()
        self.errorTypes=list(set([k[1] for k in keys]))

    def writeToError(self, error):
        new_dict = {}
        for k, v in self.dicts.items():
            if k[1]==error:
                new_dict[k] = v
        self.models = sorted(list(set([k[3] for k in new_dict.keys()])))[::-1]
        self.data = sorted(list(set([k[0] for k in new_dict.keys()])))[::-1]
        self.file= sorted(list(set([k[2] for k in new_dict.keys()])))[::-1]

    def writeToDataframe(self, output_dir):
        writer = pd.ExcelWriter(output_dir)
        for error in self.errorTypes:
            df=pd.DataFrame()
            self.writeToError(error)
            for eachdata in self.data:
                df11=pd.DataFrame()
                for eachfile in self.file:
                    values=[]
                    for eachmodel in self.models:
                        value=self.dicts.get(tuple([eachdata, error, eachfile, eachmodel]))
                        if value==None:
                            value=[value]*4
                        values.append(value)
                    flat_list = [item for sublist in values for item in sublist]
                    subrow = pd.DataFrame([flat_list])
                    df11=pd.concat([df11, subrow], axis=0, join='outer')
                df=df.append(df11)
            cols = pd.MultiIndex.from_product([self.models, ['Train', 'Val', 'Test_dirty_acc', 'Test_clean_acc', 'Test_dirty_f1', 'Test_clean_f1']])
            df.columns= cols

            index = pd.MultiIndex.from_product([self.data,self.file])
            df.index=index
            df.to_excel(writer, '%s'%error)
        writer.save()

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
# 
def combine(result):
    seeds = list({k.split('/')[4] for k in result.keys()})
    comb = {}
    for s in seeds:
        for k, v in result.items():
            dataset, error, file, model, seed = k.split('/')
            if s != seed:
                continue

            key = (dataset, error, file, model)
            value = {vk:[vv] for vk, vv in v.items()}

            if key not in comb.keys():
                comb[key] = value
            else:
                for vk, vv in v.items():
                    comb[key][vk].append(vv)
    return comb

def summarize(result, form, precision=6):
    comb = combine(result)
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

        value = [train_acc, val_acc, dirty_test_acc, clean_test_acc, dirty_test_f1, clean_test_f1]
        summary[k] = value
    return summary

result = utils.load_result()
# summary = summarize(result, form_all)
# table_generator = TableGenerator(summary)
# table_generator.writeToDataframe("all_result.xlsx")
summary = summarize(result, form_all)
table_generator = TableGenerator(summary)
table_generator.writeToDataframe("all_result.xlsx")
