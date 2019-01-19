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
result_dir = 'result.json'

# experiment
root_seed = 1
n_resplit = 20
n_retrain = 5
test_ratio = 0.3
max_size = 1000

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
    "error_types": ['duplicates'],
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
    "error_types": ['duplicates'],
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
    "drop_variables": ["Date", "Unnamed: 0", "Country", "City"]
}

datasets = [Airbnb, USCensus, Credit, EGG, Titanic, KDD,
            Marketing, Sensor, Movie, Restaurant, Citation]

# models
logistic_reg = {
    "name": "logistic_regression",
    "fn": LogisticRegression,
    "estimator": LogisticRegression(solver="liblinear", max_iter=5000, multi_class='auto'),
    "type": "classification",
    "params": "C" ,
    "params_type": "real",
    "params_range": [-5, 5]
}

knn_clf = {
    "name": "knn_classification",
    "fn": KNeighborsClassifier,
    "estimator": KNeighborsClassifier(),
    "type": "classification",
    "params": "n_neighbors",
    "params_type": "int",
    "params_range": [1, 95]
}

dt_clf = {
    "name": "decision_tree_classification",
    "fn": DecisionTreeClassifier,
    "estimator": DecisionTreeClassifier(),
    "type": "classification",
    "params": "max_depth",
    "params_type": "int",
    "params_range": [1, 200]
}

linear_svm = {
    "name": "linear_svm",
    "fn": LinearSVC,
    # "estimator": LinearSVC(max_iter=5000),
    "estimator": SVC(kernel='linear'),
    "type": "classification",
    "params": "C",
    "params_type": "real",
    "params_range": [-5, 5]
}

adaboost_clf = {
    "name": "adaboost_classification",
    "fn": AdaBoostClassifier,
    "estimator": AdaBoostClassifier(n_estimators=200),
    "type": "classification",
    "params": "learning_rate",
    "params_type": "real",
    "params_range": [-9, 1]
}

random_forest_clf = {
    "name": "random_forest_classification",
    "fn": RandomForestClassifier,
    "estimator": RandomForestClassifier(n_estimators=100),
    "type": "classification",
    "params": "max_depth",
    "params_type": "int",
    "params_range": [1, 200]
}

gaussian_nb = {
    "name": "guassian_naive_bayes",
    "fn": GaussianNB,
    "estimator": GaussianNB(),
    "type": "classification"
}

models = [logistic_reg, knn_clf, dt_clf, adaboost_clf, random_forest_clf, gaussian_nb]