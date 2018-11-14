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
        self.models = list(set([k[3] for k in new_dict.keys()]))
        self.data = list(set([k[0] for k in new_dict.keys()]))
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
                        value=self.dicts.get(tuple([eachdata,error, eachfile, eachmodel]))
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

res = utils.load_result()
for ml_type in ["classification"]:
    result = {}
    for k, v in res.items():
        dataset, error, file, model, seed = k.split('/')
        key = (dataset, error, file, model, seed)

        train_acc = "{:.6f}".format(v['train_acc'])
        val_acc = "{:.6f}".format(v['val_acc'])
        dirty_test_acc = "{:.6f}".format(v['dirty_test_acc'])
        dirty_test_f1 = "{:.6f}".format(v['dirty_test_f1'])
        if error == 'missing_values' and file == 'dirty':
            clean_test_acc = "{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}".format(v["clean_impute_mode_mode_test_acc"], 
                                                                             v["clean_impute_mode_dummy_test_acc"], 
                                                                             v["clean_impute_median_mode_test_acc"], 
                                                                             v["clean_impute_median_dummy_test_acc"], 
                                                                             v["clean_impute_mean_mode_test_acc"], 
                                                                             v["clean_impute_mean_dummy_test_acc"])
            clean_test_f1 = "{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}".format(v["clean_impute_mode_mode_test_f1"], 
                                                             v["clean_impute_mode_dummy_test_f1"], 
                                                             v["clean_impute_median_mode_test_f1"], 
                                                             v["clean_impute_median_dummy_test_f1"], 
                                                             v["clean_impute_mean_mode_test_f1"], 
                                                             v["clean_impute_mean_dummy_test_f1"])
        elif error == 'outliers' and file == 'dirty':
            clean_test_acc = "{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}".format(v["clean_iso_forest_impute_median_dummy_test_acc"],
                                                                             v["clean_IQR_impute_mean_dummy_test_acc"], 
                                                                             v["clean_iso_forest_delete_test_acc"], 
                                                                             v["clean_SD_impute_median_dummy_test_acc"],
                                                                             v["clean_iso_forest_impute_mean_dummy_test_acc"],
                                                                             v["clean_SD_delete_test_acc"], 
                                                                             v["clean_IQR_impute_median_dummy_test_acc"],
                                                                             v["clean_IQR_impute_mean_dummy_test_acc"],
                                                                             v["clean_IQR_delete_test_acc"])
            clean_test_f1 = "{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}/{:.6f}".format(v["clean_iso_forest_impute_median_dummy_test_f1"],
                                                             v["clean_IQR_impute_mean_dummy_test_f1"], 
                                                             v["clean_iso_forest_delete_test_f1"], 
                                                             v["clean_SD_impute_median_dummy_test_f1"],
                                                             v["clean_iso_forest_impute_mean_dummy_test_f1"],
                                                             v["clean_SD_delete_test_f1"], 
                                                             v["clean_IQR_impute_median_dummy_test_f1"],
                                                             v["clean_IQR_impute_mean_dummy_test_f1"],
                                                             v["clean_IQR_delete_test_f1"])
        else:
            clean_test_acc = "{:.6f}".format(v[file + '_test_acc'])
            clean_test_f1 = "{:.6f}".format(v[file + '_test_f1'])

        value = [train_acc, val_acc, dirty_test_acc, clean_test_acc, dirty_test_f1, clean_test_f1]

        if utils.get_model(model)['type'] == ml_type:
            result[key] = value
    seeds = list({seed for dataset, error, file, model, seed in result.keys()})

    combine = {}
    for s in seeds:
        for k, v in result.items():
            dataset, error, file, model, seed = k
            if seed != s:
                continue
            key = (dataset, error, file, model)
            if key not in combine.keys():
                combine[key] = v
            else:
                for i, acc in enumerate(v):
                    combine[key][i] += ", {}".format(acc) 

    table_generator = TableGenerator(combine)
    table_generator.writeToDataframe("{}_result.xlsx".format(ml_type))

