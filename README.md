# CleaningForML
This repository provides the implementation of experiments as described in the paper:
> Impacts of Dirty Data on Machine Learning Models

### Basic Usage
To run experiments, execute the following command from the project home directory:

```
python main.py [--dataset <name>] [--cpu <num_cpu>] 
```

#### Options:
--dataset: specify the experiment dataset. If not specified, the program will run experiments on all datasets.
--cpu: specify the number of cpu used for experiment. Default is 1.

#### Output:
The experimental results for each dataset will be saved in /result directory as a json file named as \<dataset name\>\_result.json. Each result is a key-value pair. The key is a string in format "\<dataset\>/\<split seed\>/\<error type\>/\<clean method\>/\<ML model\>/\<experiment seed\>". The value is a set of key-value pairs for each evaluation metric and result.

To compare the results and perform hypothesis test, execute the following command from the project home directory:

```
python table.py
```

#### Output:
The results of comparison and hypothesis test will be saved in /table directory.










