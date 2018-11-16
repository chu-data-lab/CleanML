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
    "categorical_variables": ['Rating'],
    "ml_task": "classification",
    'key_columns': ['latitude', 'longitude'],
    "manual_clean_duplicates": False
}

DfD = {
    "data_dir": "DfD",
    "error_types": ['inconsistency'],
    "categorical_variables":[],
    "label": "Donate",
    "ml_task": "classification"
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
    "error_types": ['duplicates','inconsistency'],
    "label": "priceRange",
    "ml_task": "classification",
    "drop_variables": ["streetAddress", "telephone", "website"],
    "text_variables": ["name", "categories", "neighborhood"],
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
    "ml_task": "classification",
    "manual_clean_duplicates": False
}

Food = {
    "data_dir": "Food",
    "error_types": ['missing_values', 'inconsistency'],
    "categorical_variables": ['Violations', "Results"],
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
            Marketing, Sensor, Movie, Food, Restaurant]

# Citation, Restaurant,Food, Movie

