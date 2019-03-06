"""Define the domain of dataset"""
from .error_type import *

# details of each dataset
KDD = {
    "data_dir": "KDD",
    "error_types": ["missing_values", "outliers"],
    "label": 'is_exciting_20',
    "ml_task": "classification",
    "class_imbalance": True,
    "categorical_variables":['is_exciting_20'],
}

Citation = {
    "data_dir": "Citation",
    "error_types": ["duplicates"],
    'key_columns': ['title'],
    "label":"CS",
    "ml_task": "classification",
    "text_variables":["title"],
}

Marketing = {
    "data_dir": "Marketing",
    "error_types": ["missing_values"],
    "label": 'Income',
    "ml_task": "classification"
}

Airbnb = {
    "data_dir": "Airbnb",
    "error_types": ["missing_values", "outliers", "duplicates"],
    "label": 'Rating',
    "categorical_variables": ['Rating'],
    "ml_task": "classification",
    'key_columns': ['latitude', 'longitude'],
}

Titanic = {
    "data_dir": "Titanic",
    "error_types": ["missing_values"],
    "drop_variables": ['PassengerId', 'Name'],
    "label": "Survived",
    "categorical_variables":["Survived"],
    "ml_task": "classification"
}

EEG = {
    "data_dir": "EEG",
    "error_types": ["outliers"],
    'label':'Eye',
    "categorical_variables":['Eye'],
    "ml_task": "classification"
}

USCensus = {
    "data_dir": "USCensus",
    "error_types": ["missing_values"],
    "label": 'Income',
    "ml_task": "classification"
}

Restaurant = {
    "data_dir": "Restaurant",
    "error_types": ["duplicates", "inconsistency"],
    "label": "priceRange",
    "ml_task": "classification",
    "drop_variables": ["streetAddress", "telephone", "website"],
    "text_variables": ["name", "categories", "neighborhood"],
    "key_columns": ["telephone"]
}

Credit = {
    "data_dir": "Credit",
    "error_types": ["missing_values", "outliers"],
    "label": "SeriousDlqin2yrs",
    "categorical_variables":["SeriousDlqin2yrs"],
    "ml_task": "classification",
    "class_imbalance":True
}

Sensor = {
    "data_dir": "Sensor",
    "error_types": ["outliers"],
    "categorical_variables": ['moteid'],
    "label": 'moteid',
    "ml_task": "classification"
}

Movie = {
    "data_dir": "Movie",
    "error_types": ["duplicates", "inconsistency"],
    "key_columns": ["title", "year"],
    "categorical_variables": ["genres"],
    "text_variables": ["title"],
    "label": "genres",
    "ml_task": "classification"
}

Company = {
    "data_dir": "Company",
    "error_types": ["inconsistency"],
    "label": "Sentiment",
    "ml_task": "classification",
    "drop_variables": ["Date", "Unnamed: 0", "City"]
}

University = {
    "data_dir": "University",
    "error_types": ["inconsistency"],
    "label": "expenses thous$",
    "ml_task": "classification",
    "drop_variables": ["university name", "academic-emphasis"]
}

KDD_major = {
    "data_dir": "KDD_major",
    "error_types": ["mislabel"],
    "label": 'is_exciting_20',
    "ml_task": "classification",
    "class_imbalance": True,
    "categorical_variables":['is_exciting_20'],
}

KDD_minor = {
    "data_dir": "KDD_minor",
    "error_types": ["mislabel"],
    "label": 'is_exciting_20',
    "ml_task": "classification",
    "class_imbalance": True,
    "categorical_variables":['is_exciting_20'],
}

KDD_uniform = {
    "data_dir": "KDD_uniform",
    "error_types": ["mislabel"],
    "label": 'is_exciting_20',
    "ml_task": "classification",
    "class_imbalance": True,
    "categorical_variables":['is_exciting_20'],
}

USCensus_major = {
    "data_dir": "USCensus_major",
    "error_types": ["mislabel"],
    "label": 'Income',
    "ml_task": "classification"
}

USCensus_minor = {
    "data_dir": "USCensus_minor",
    "error_types": ["mislabel"],
    "label": 'Income',
    "ml_task": "classification"
}

USCensus_uniform = {
    "data_dir": "USCensus_uniform",
    "error_types": ["mislabel"],
    "label": 'Income',
    "ml_task": "classification"
}

EEG_major = {
    "data_dir": "EEG_major",
    "error_types": ["mislabel"],
    'label':'Eye',
    "categorical_variables":['Eye'],
    "ml_task": "classification"
}

EEG_minor = {
    "data_dir": "EEG_minor",
    "error_types": ["mislabel"],
    'label':'Eye',
    "categorical_variables":['Eye'],
    "ml_task": "classification"
}

EEG_uniform = {
    "data_dir": "EEG_uniform",
    "error_types": ["mislabel"],
    'label':'Eye',
    "categorical_variables":['Eye'],
    "ml_task": "classification"
}

# domain of dataset 
datasets = [KDD, Credit, Airbnb, USCensus, EEG, Titanic, 
            Marketing, Sensor, Movie, Restaurant, Citation, 
            Company, University, KDD_uniform, KDD_minor, KDD_major,
            USCensus_uniform, USCensus_major, USCensus_minor,
            EEG_uniform, EEG_minor, EEG_major]