# Datasets
root_dir = 'datasets'

KDD = {
    "data_dir": "KDD",
    "error_types": ['mv', 'out'],
    "label": 'is_exciting_20',
    "categorical_variables":['is_exciting_20']
}

Citation = {
    "data_dir": "Citation",
    "error_types": ['mv', 'dup'],
    "categorical_variables":[],
    "dup_ground_truth": True
}

Marketing = {
    "data_dir": "Marketing",
    "error_types": ['mv'],
    "label": 'Income',
    "categorical_variables":[]
}

Airbnb = {
    "data_dir": "Airbnb",
    "error_types": ['mv', 'out', 'dup'],
    "label": 'Rating',
    "categorical_variables":[],
    'key_columns': ['latitude', 'longitude'],
    "dup_ground_truth": False
}

DfD = {
    "data_dir": "DfD",
    "error_types": ['incon', 'mislabel'],
    "categorical_variables":[]
}

Titanic = {
    "data_dir": "Titanic",
    "error_types": ['mv'],
    "label": "Survived",
    "categorical_variables":["Survived"]
}

EGG = {
    "data_dir": "EGG",
    "error_types": ['out'],
    'label':'Eye',
    "categorical_variables":['Eye']
}

USCensus = {
    "data_dir": "USCensus",
    "error_types": ['mv'],
    "label": 'Income',
    "categorical_variables":[]
}

Restaurant = {
    "data_dir": "Restaurant",
    "error_types": ['mv', 'dup','incon'],
    "categorical_variables":[],
    "key_columns": ["telephone"],
    "dup_ground_truth": False
}

Credit = {
    "data_dir": "Credit",
    "error_types": ['mv', 'out'],
    "label": "SeriousDlqin2yrs",
    "categorical_variables":["SeriousDlqin2yrs"]
}

Sensor = {
    "data_dir": "Sensor",
    "error_types": ['mv', 'out'],
    "categorical_variables": ['moteid']
}

Movie = {
    "data_dir": "Movie",
    "error_types": ['dup', 'incon'],
    "key_columns": ["title", "year"],
    "dup_ground_truth": False,
    "categorical_variables": []
}

datasets = [KDD, Citation, Marketing, Airbnb, DfD, Titanic, 
        EGG, USCensus, Restaurant, Credit, Sensor, Movie]



