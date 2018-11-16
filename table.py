import json
import pandas as pd
import numpy as np
import utils
import re

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

def get_result(result_dict, key, precision=6):
    if key in result_dict.keys():
        s = "{1:.{0}f}".format(precision, result_dict[key])
    else:
        s = "nan"
    return s

def summarize(result, precision=6):
    seeds = list({k.split('/')[4] for k in result.keys()})
    summary = {}
    for s in seeds:
        for k, v in result.items():
            dataset, error, file, model, seed = k.split('/')
            if s != seed:
                continue

            key = (dataset, error, file, model)
            test_files = sorted(utils.get_test_files(error, file))[::-1]
            test_acc_keys = ['{}_test_acc'.format(f) for f in test_files if f[0] == 'c']
            test_f1_keys = ['{}_test_f1'.format(f) for f in test_files if f[0] == 'c']

            train_acc = get_result(v, 'train_acc')
            val_acc = get_result(v, 'val_acc')
            dirty_test_acc = get_result(v, 'dirty_test_acc')
            dirty_test_f1 = get_result(v, 'dirty_test_f1')
            clean_test_acc = "/".join([get_result(v, test_k) for test_k in test_acc_keys])
            clean_test_f1 = "/".join([get_result(v, test_k) for test_k in test_f1_keys])

            value = [train_acc, val_acc, dirty_test_acc, clean_test_acc, dirty_test_f1, clean_test_f1]

            if key not in summary.keys():
                summary[key] = value
            else:
                for i, acc in enumerate(summary[key]):
                    summary[key][i] += ", {}".format(value[i]) 
    return summary

result = utils.load_result()
summary = summarize(result)
table_generator = TableGenerator(summary)
table_generator.writeToDataframe("result.xlsx")

