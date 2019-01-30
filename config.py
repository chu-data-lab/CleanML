"""Configuration of experiment, datasets and models"""
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RANSACRegressor

# directory
data_dir = 'data'
result_dir = 'result'
table_dir = 'table'

# experiment
root_seed = 1
n_resplit = 20
n_retrain = 5
test_ratio = 0.3
max_size = None

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
    'key_columns': ['title'],
    "label":"CS",
    "ml_task": "classification",
    "text_variables":["title"],
}

Marketing = {
    "data_dir": "Marketing",
    "error_types": ['missing_values'],
    "label": 'Income',
    "ml_task": "classification"
}

Airbnb = {
    "data_dir": "Airbnb",
    "error_types": ['missing_values', 'outliers', 'duplicates', 'mislabel'],
    "label": 'Rating',
    "categorical_variables": ['Rating'],
    "ml_task": "classification",
    'key_columns': ['latitude', 'longitude'],
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
    "error_types": ['duplicates', 'inconsistency'],
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
    "error_types": ['duplicates', 'inconsistency'],
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

datasets = [KDD, Credit, Airbnb, USCensus, EGG, Titanic, Marketing, Sensor, Movie, Restaurant, Citation, Company, University]

# models
logistic_reg = {
    "name": "logistic_regression",
    "fn": LogisticRegression,
    "fixed_params": {"solver":"lbfgs", "max_iter":5000, "multi_class":'auto'},
    "parallelable": True,
    "type": "classification",
    "hyperparams": "C" ,
    "hyperparams_type": "real",
    "hyperparams_range": [-5, 5]
}

knn_clf = {
    "name": "knn_classification",
    "fn": KNeighborsClassifier,
    "fixed_params":{},
    "parallelable": True,
    "type": "classification",
    "hyperparams": "n_neighbors",
    "hyperparams_type": "int",
    "hyperparams_range": [1, 95]
}

dt_clf = {
    "name": "decision_tree_classification",
    "fn": DecisionTreeClassifier,
    "fixed_params": {},
    "type": "classification",
    "hyperparams": "max_depth",
    "hyperparams_type": "int",
    "hyperparams_range": [1, 200]
}

linear_svm = {
    "name": "linear_svm",
    # "fn": LinearSVC,
    # "estimator": LinearSVC(max_iter=5000),
    "fn": SVC,
    "fixed_params": {"kernel":"linear", "cache_size":7000},
    "type": "classification",
    "hyperparams": "C",
    "hyperparams_type": "real",
    "hyperparams_range": [-5, 5]
}

adaboost_clf = {
    "name": "adaboost_classification",
    "fn": AdaBoostClassifier,
    "fixed_params": {"n_estimators":200},
    "type": "classification",
    "hyperparams": "learning_rate",
    "hyperparams_type": "real",
    "hyperparams_range": [-9, 1]
}

random_forest_clf = {
    "name": "random_forest_classification",
    "fn": RandomForestClassifier,
    "fixed_params": {"n_estimators":100},
    "parallelable": True,
    "type": "classification",
    "hyperparams": "max_depth",
    "hyperparams_type": "int",
    "hyperparams_range": [1, 200]
}

gaussian_nb = {
    "name": "guassian_naive_bayes",
    "fn": GaussianNB,
    "fixed_params": {},
    "type": "classification"
}

models = [logistic_reg, knn_clf, dt_clf, adaboost_clf, random_forest_clf, gaussian_nb, linear_svm]