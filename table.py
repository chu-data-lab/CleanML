import json
import pandas as pd
import numpy as np
import utils
import re

class TableGenerator(object):
    def __init__(self, results):
        self.results = results
        self.dicts={tuple(k.split('/')): v.split('/') for k, v in self.results.items()}
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
            cols = pd.MultiIndex.from_product([self.models, ['Train', 'Val', 'Test_dirty', 'Test_clean']])
            df.columns= cols

            index = pd.MultiIndex.from_product([self.data,self.file])
            df.index=index
            df.to_excel(writer, '%s'%error)
            writer.save()

res = utils.load_result()
for ml_type in ["classification", "regression"]:
    result = {}
    for k, v in res.items():
        dataset, error, file, model = k.split('/')
        param, train, val, test = v
        if file[0] == "d":
            value = "{:.4f}/{:.4f}/{:.4f}/".format(train, val, test)
        else:
            value = "{:.4f}/{:.4f}//{:.4f}".format(train, val, test)
        if utils.get_model(model)['type'] == ml_type:
            result[k] = value

    table_generator = TableGenerator(result)
    table_generator.writeToDataframe("{}_result.xlsx".format(ml_type))

