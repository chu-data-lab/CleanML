from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RANSACRegressor

linear_reg = {
    "name": "linear_regression",
    "fn": Lasso,
    "type": "regression",
    "params": "alpha",
    "params_space": "log"
}

logistic_reg = {
    "name": "logistic_regression",
    "fn": LogisticRegression,
    "type": "classification",
    "params": "C" ,
    "params_space": "log"
}

knn_clf = {
    "name": "knn_classification",
    "fn": KNeighborsClassifier,
    "type": "classification",
    "params": "n_neighbors",
    "params_space": "int"
}

knn_reg = {
    "name": "knn_regression",
    "fn": KNeighborsRegressor,
    "type": "regression",
    "params": "n_neighbors",
    "params_space": "int"
}

dt_reg = {
    "name": "decision_tree_regression",
    "fn": DecisionTreeRegressor,
    "type": "regression",
    "params": "max_depth",
    "params_space": "int"
}

dt_clf = {
    "name": "decision_tree_classification",
    "fn": DecisionTreeRegressor,
    "type": "classification",
    "params": "max_depth",
    "params_space": "int"
}

linear_svm = {
    "name": "linear_svm",
    "fn": LinearSVC,
    "type": "classification",
    "params": "C",
    "params_space": "log"
}

adaboost = {
    "name": "adaboost",
    "fn": AdaBoostClassifier,
    "type": "classification",
    "params": "learning_rate",
    "params_space": "log"
}

models = [linear_reg, logistic_reg, knn_clf, knn_reg, linear_svm, dt_reg, dt_clf, adaboost]
