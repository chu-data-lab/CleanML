from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RANSACRegressor

linear_reg = {
    "name": "linear_regression",
    "fn": Lasso,
    "estimator": Lasso(),
    "type": "regression",
    "params": "alpha",
    "params_type": "real"
}

logistic_reg = {
    "name": "logistic_regression",
    "fn": LogisticRegression,
    "estimator": LogisticRegression(solver="liblinear", max_iter=10000, multi_class='auto'),
    "type": "classification",
    "params": "C" ,
    "params_type": "real"
}

knn_clf = {
    "name": "knn_classification",
    "fn": KNeighborsClassifier,
    "estimator": KNeighborsClassifier(),
    "type": "classification",
    "params": "n_neighbors",
    "params_type": "int"
}

knn_reg = {
    "name": "knn_regression",
    "fn": KNeighborsRegressor,
    "estimator": KNeighborsRegressor(),
    "type": "regression",
    "params": "n_neighbors",
    "params_type": "int"
}

dt_reg = {
    "name": "decision_tree_regression",
    "fn": DecisionTreeRegressor,
    "estimator": DecisionTreeRegressor(),
    "type": "regression",
    "params": "max_depth",
    "params_type": "int"
}

dt_clf = {
    "name": "decision_tree_classification",
    "fn": DecisionTreeClassifier,
    "estimator": DecisionTreeClassifier(),
    "type": "classification",
    "params": "max_depth",
    "params_type": "int"
}

linear_svm = {
    "name": "linear_svm",
    "fn": LinearSVC,
    "estimator": LinearSVC(max_iter=10000),
    "type": "classification",
    "params": "C",
    "params_type": "real"
}

adaboost_clf = {
    "name": "adaboost_classification",
    "fn": AdaBoostClassifier,
    "estimator": AdaBoostClassifier(),
    "type": "classification",
    "params": "learning_rate",
    "params_type": "real"
}

adaboost_reg = {
    "name": "adaboost_regression",
    "fn": AdaBoostRegressor,
    "estimator": AdaBoostRegressor(),
    "type": "regression",
    "params": "learning_rate",
    "params_type": "real"
}

random_forest_clf = {
    "name": "random_forest_classification",
    "fn": RandomForestClassifier,
    "estimator": RandomForestClassifier(),
    "type": "classification",
    "params": "max_depth",
    "params_type": "int"
}

random_forest_reg = {
    "name": "random_forest_regression",
    "fn": RandomForestRegressor,
    "estimator": RandomForestRegressor(),
    "type": "regression",
    "params": "max_depth",
    "params_type": "int"
}

gaussian_nb = {
    "name": "guassian_naive_bayes",
    "fn": GaussianNB,
    "estimator": GaussianNB(),
    "type": "classification"
}

models = [linear_reg, logistic_reg, knn_clf, knn_reg, linear_svm, dt_reg, dt_clf, adaboost_reg, adaboost_clf,
          random_forest_reg, random_forest_clf, gaussian_nb]
