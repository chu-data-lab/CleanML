"""Define the domain of ML model"""
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
from xgboost import XGBClassifier

# details of each model
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

xgb_clf = {
    "name":"XGBoost",
    "fn": XGBClassifier,
    "fixed_params": {},
    "type": "classification",
    "hyperparams": "max_depth",
    "hyperparams_type": "int",
    "hyperparams_range": [1, 100]
}

# model domain
models = [logistic_reg, knn_clf, dt_clf, adaboost_clf, random_forest_clf, gaussian_nb, xgb_clf]
