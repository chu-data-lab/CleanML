"""Configuration"""

# directory
root_dir = 'datasets'
result_dir = 'result.json'

# datasets
KDD = {
    "data_dir": "KDD",
    "error_types": ['missing_values', 'outliers', 'mislabel'],
    "label": 'is_exciting_20',
    "ml_task": "classification",
    "class_imbalance": True,
    "categorical_variables":['is_exciting_20'],
}

Citation = {
    "data_dir": "Citation",
    "error_types": ['duplicates'],
    'key_columns': ['titleWOS', 'yearWOS'],
    "label":"CS",
    "ml_task": "classification",
    "text_variables":["titleWOS", "venueWOS", "keywords", "abstract"],
}

Marketing = {
    "data_dir": "Marketing",
    "error_types": ['missing_values'],
    "label": 'Income',
    "ml_task": "classification"
}

Airbnb = {
    "data_dir": "Airbnb",
    "error_types": ['missing_values', 'outliers', 'duplicates'],
    "label": 'Rating',
    "categorical_variables": ['Rating'],
    "ml_task": "classification",
    'key_columns': ['latitude', 'longitude'],
}

DfD = {
    "data_dir": "DfD",
    "error_types": ['inconsistency'],
    "text_variables":["Name1"],
    "label": "Donation",
    "ml_task": "classification",
    "class_imbalance": True
}

Titanic = {
    "data_dir": "Titanic",
    "error_types": ['missing_values'],
    "drop_variables": ['PassengerId', 'Name'],
    "label": "Survived",
    "categorical_variables":["Survived"],
    "ml_task": "classification"
}

EGG = {
    "data_dir": "EGG",
    "error_types": ['outliers', 'mislabel'],
    'label':'Eye',
    "categorical_variables":['Eye'],
    "ml_task": "classification"
}

USCensus = {
    "data_dir": "USCensus",
    "error_types": ['missing_values', 'mislabel'],
    "label": 'Income',
    "ml_task": "classification"
}

Restaurant = {
    "data_dir": "Restaurant",
    "error_types": ['duplicates','inconsistency'],
    "label": "priceRange",
    "ml_task": "classification",
    "drop_variables": ["streetAddress", "telephone", "website"],
    "text_variables": ["name", "categories", "neighborhood"],
    "key_columns": ["telephone"]
}

Credit = {
    "data_dir": "Credit",
    "error_types": ['missing_values', 'outliers'],
    "label": "SeriousDlqin2yrs",
    "categorical_variables":["SeriousDlqin2yrs"],
    "ml_task": "classification",
    "class_imbalance":True
}

Sensor = {
    "data_dir": "Sensor",
    "error_types": ['outliers'],
    "categorical_variables": ['moteid'],
    "label": 'moteid',
    "ml_task": "classification"
}

Movie = {
    "data_dir": "Movie",
    "error_types": ['inconsistency', 'duplicates'],
    "key_columns": ["title", "year"],
    "categorical_variables": ["genres"],
    "text_variables": ["title"],
    "label": "genres",
    "ml_task": "classification"
}

Food = {
    "data_dir": "Food",
    "error_types": ['inconsistency'],
    "categorical_variables": ['Violations', "Results"],
    "drop_variables": ["Inspection Date"],
    "label": "Results",
    "ml_task": "classification",
    "drop_variables":[],
    "text_variables":["DBA Name"]
}

IMDB = {
    "data_dir": "IMDB",
    "error_types": ['mislabel'],
    "label": 'genres',
    "ml_task": "classification"
}

datasets = [Airbnb, USCensus, Credit, EGG, Titanic, KDD,
            Marketing, Sensor, Movie, Food, Restaurant, DfD]