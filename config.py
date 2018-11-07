# Datasets
root_dir = 'datasets'
result_dir = 'result.json'

KDD = {
    "data_dir": "KDD",
    "error_types": ['missing_values', 'outliers'],
    "label": 'is_exciting_20',
    "ml_task": "classification",
    "class_imbalance": True,
    "categorical_variables":['is_exciting_20'],
}

Citation = {
    "data_dir": "Citation",
    "error_types": ['missing_values', 'duplicates', 'inconsistency'],
    "label":"venue",
    "ml_task": "classification",
    "text_variables":["title", "authors"],
    "drop_variables":["id"],
    "manual_clean_duplicates": True
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
    "ml_task": "regression",
    'key_columns': ['latitude', 'longitude'],
    "manual_clean_duplicates": False
}

DfD = {
    "data_dir": "DfD",
    "error_types": ['inconsistency', 'mislabel'],
    "categorical_variables":[],
    "label": "Donate",
    "ml_task": "classification"
}

Titanic = {
    "data_dir": "Titanic",
    "error_types": ['missing_values'],
    "label": "Survived",
    "categorical_variables":["Survived"],
    "ml_task": "classification"
}

EGG = {
    "data_dir": "EGG",
    "error_types": ['outliers'],
    'label':'Eye',
    "categorical_variables":['Eye'],
    "ml_task": "classification"
}

USCensus = {
    "data_dir": "USCensus",
    "error_types": ['missing_values'],
    "label": 'Income',
    "ml_task": "classification"
}

Restaurant = {
    "data_dir": "Restaurant",
    "error_types": ['missing_values', 'duplicates','inconsistency'],
    "label": "priceRange",
    "ml_task": "classification",
    "text_variables": ["name", "streetAddress", "telephone", "website"],
    "key_columns": ["telephone"],
    "manual_clean_duplicates": False
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
    "label": 'temperature',
    "ml_task": "regression"
}

Movie = {
    "data_dir": "Movie",
    "error_types": ['duplicates', 'inconsistency'],
    "key_columns": ["title", "year"],
    "categorical_variables": ["genres"],
    "text_variables": ["title"],
    "label": "genres",
    "ml_task": "classification",
    "manual_clean_duplicates": False
}

Food = {
    "data_dir": "Food",
    "error_types": ['inconsistency'],
    "categorical_variables": [],
    "label": "Results",
    "ml_task": "classification",
    "drop_variables":["Inspection ID", "Location", 'Address', 'License #', "AKA Name"],
}

datasets = [Airbnb, USCensus, Credit, EGG, Titanic, KDD,
            Marketing, Sensor]

# Citation, Restaurant,Food, Movie

