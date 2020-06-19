# CleanML

This is the CleanML Benchmark for Joint Data Cleaning and Machine Learning. 

The codebase is located in: https://github.com/chu-data-lab/CleanML

The details of the benchmark methodology and design are described in the paper:
> CleanML: A Benchmark for
Joint Data Cleaning and Machine Learning [Experiments and Analysis]



## Basic Usage
### Run Experiments
To run experiments, download and unzip the [datasets](https://www.dropbox.com/s/62xlcfsoykl2k7n/CleanML-datasets-2020.zip?dl=0). Place it under the project home directory and execute the following command from the project home directory:

```
python3 main.py --run_experiments [--dataset <name>] [--cpu <num_cpu>] [--log]
```

#### Options:
--dataset: the experiment dataset. If not specified, the program will run experiments on all datasets.<br>
--cpu: the number of cpu used for experiment. Default is 1.<br>
--log: whether to log experiment process

#### Output:
The experimental results for each dataset will be saved in `/result` directory as a json file named as \<dataset name\>\_result.json. Each result is a key-value pair. The key is a string in format "\<dataset\>/\<split seed\>/\<error type\>/\<clean method\>/\<ML model\>/\<random search seed\>". The value is a set of key-value pairs for each evaluation metric and result. Our experimental results are provided in `result.zip`.

### Run Analysis
To run analysis for populating relations described in the paper, unzip `result.zip` and execute the following command from the project home directory:

```
python3 main.py --run_analysis [--alpha <value>]
```

#### Options:
--alpha: the significance level for multiple hypothesis test. Default is 0.05.

#### Output:
The relations R1, R2 and R3 will be saved in `/analysis` directory. Our analysis results are provided in `analysis.zip`.

## Extend Domain of Attributes
### Add new datasets:
To add a new dataset, first, create a new folder with dataset name under `/data` and create a `raw` folder under the new folder.  The `raw` folder must contain raw data named `raw.csv`. For dataset with inconsistencies, it must also contain the inconsistency-cleaned version data named `inconsistency_clean_raw.csv`. For dataset with mislabels, it must also contain the mislabel-cleaned version data named `mislabel_clean_raw.csv`. The structure of the directory looks like:
<pre>
.
└── data
    └── new_dataset
        └── raw
            ├── raw.csv
            ├── inconsistency_clean_raw.csv (for dataset with inconsistencies)
            └── mislabel_clean_raw.csv (for dataset with mislabels)
</pre>

Then add a dictionary to `/schema/dataset.py` and append it to `datasets` array at the end of the file.<br> 

The new dictionary must contain the following keys:<br>
```yaml
data_dir: the name of the dataset.
error_types: a list of error types that the dataset contains.
label: the label of ML task.
```

The following keys are optional:<br>
```yaml
class_imbalance: whether the dataset is class imbalanced.
categorical_variables: a list of categorical attributes.
text_variables: a list of text attributes.
key_columns: a list of key columns used for deduplication.
drop_variables: a list of irrelevant attributes.
```
### Add new error types:
To add a new error type, add a dictionary to `/schema/error_type.py` and append it to `error_types` array at the end of the file. <br>

The new dictionary must contain the following keys:<br>
```yaml
name: the name of the error type.
cleaning_methods: a dictionary, {cleaning method name: cleaning methods object}.
```
### Add new models:
To add a new ML model, add a dictionary to `/schema/model.py` and append it to `models` array at the end of the file. <br>

The new dictionary must contain the following keys:<br>
```yaml
name: the name of the model.
fn: the function of the model.
fixed_params: parameters not to be tuned.
hyperparams: the hyperparameter to be tuned.
hyperparams_type: the type of hyperparameter "real" or "int".
hyperparams_range: range of search. Use log base for real type hyperparameters.
```
### Add new cleaning methods:
To add a new cleaning methods, add a class to `/schema/cleaning_method.py`. <br>

The class must contain two methods:<br>

`fit(dataset, dirty_train)`: take in the dataset dictionary and dirty training set. Compute statistics or train models on training set for data cleaning.<br>
`clean(dirty_train, dirty_test)`: take in the dirty training set and dirty test set. Clean the error in the training set and test set. Return `(clean_train, indicator_train, clean_test, indicator_test)`, which are the clean version datasets and indicators that indicate the location of error. 

### Add new scenarios:
We consider "BD" and "CD" scenarios in our paper. To investigate other scenarios, add scenarios to `/schema/scenario.py`.
