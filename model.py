from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
    "range_space": "log"
}

logistic_reg = {
    "name": "logistic_regression",
    "fn": LogisticRegression,
    "type": "classification",
    "params": "C" ,
    "range_space": "log"
}

knn_clf = {
    "name": "knn_classification",
    "fn": KNeighborsClassifier,
    "type": "classification",
    "params": "n_neighbors",
    "range_space": "int"
}

knn_reg = {
    "name": "knn_regression",
    "fn": KNeighborsRegressor,
    "type": "regression",
    "params": "n_neighbors",
    "range_space": "int"
}

linear_svm = {
    "name": "linear_svm",
    "fn": LinearSVC,
    "type": "classification",
    "params": "C",
    "range_space": "log"
}

adaboost = {
    "name": "adaboost",
    "fn": AdaBoostClassifier,
    "type": "classification",
    "params": "learning_rate",
    "range_space": "log"
}

models = [linear_reg, logistic_reg, knn_clf, knn_reg, linear_svm]
