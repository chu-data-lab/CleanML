# CleaningForML
Data Cleaning for Machine Learning

## Data Cleaning
To clean the data:
'''
python clean.py error_type [--dataset <name>]
'''

error_type: 
    --mv:   Missing values
    --out:  Outliers
    --dup:  Duplicates

If --dataset is missing, all datasets containing the error type will be cleaned.

Example: 
'''
python clean.py --mv --dataset KDD
'''

