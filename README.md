# CleanML
This repository provides the implementation of experiments as described in the paper:
> CleanML: A Benchmark for
Joint Data Cleaning and Machine Learning [Experiments and Analysis]

## Basic Usage
### Run Experiments
To run experiments, execute the following command from the project home directory:

```
python main.py --run_experiments [--dataset <name>] [--cpu <num_cpu>] [--log]
```

#### Options:
--dataset: the experiment dataset. If not specified, the program will run experiments on all datasets.<br>
--cpu: the number of cpu used for experiment. Default is 1.<br>
--log: whether to log experiment process

#### Output:
The experimental results for each dataset will be saved in /result directory as a json file named as \<dataset name\>\_result.json. Each result is a key-value pair. The key is a string in format "\<dataset\>/\<split seed\>/\<error type\>/\<clean method\>/\<ML model\>/\<random search seed\>". The value is a set of key-value pairs for each evaluation metric and result.

### Run Analysis
To run analysis for populating relations described in the paper, execute the following command from the project home directory:

```
python main.py --analysis [--alpha <name>]
```

#### Options:
--alpha: the significance level for multiple hypothesis test. Default is 0.05.

#### Output:
The relations R1, R2 and R3 will be saved in /analysis directory.

## Extend Domain of Attributes
#### Add new datasets:
To add a new dataset, add a dictionary to /schema/dataset.py and append it to datasets array at the end of the file.<br> 
The new dictionary must contain the following keys:<br>
data_dir: the name of the dataset<br>
error_types: error types that the dataset contains<br>
label: label of the dataset

The following keys are optional:<br>
class_imbalance: whether the dataset has is class imbalanced.<br>

#### Add new error types:
To add a new error type, add a dictionary to /schema/error_type.py and append it to error_types array at the end of the file. <br>
The new dictionary must contain the following keys:<br>
name: the name of the error type<br>
cleaning methods: a dictionary, where keys are names of cleaning methods, values are cleaning methods object<br>

#### Add new models:
To add a new ML model, add a dictionary to /schema/model.py and append it to models array at the end of the file. <br>
The new dictionary must contain the following keys:<br>
name: the name of the model.<br>
fn: the function of the model.<br>
fixed_params: fixed parameters during hyperparameter tuning.<br>
hyperparams: the hyperparameter to tune.<br>
hyperparams_type: the type of hyperparameter "real" or "int".<br>
hyperparams_range: range of search. Use log base for real type hyperparameter.<br>

#### Add new cleaning methods:
To add a new cleaning methods, add a class to /schema/cleaning_method.py. The class must contain two methods:<br>
fit(dataset, train): compute statistics or train models on training set for data cleaning<br>
clean(dirty_train, dirty_test): clean the error in the training set and test set. Return (clean_train, indicator_train, clean_test, indicator_test), which are the clean version dataset and indicators that indicate the location of error. 