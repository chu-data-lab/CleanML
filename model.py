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
    "estimator": Lasso(),
    "type": "regression",
    "params": "alpha",
    "params_space": "log"
}

logistic_reg = {
    "name": "logistic_regression",
    "estimator": LogisticRegression(solver="lbfgs", max_iter=100),
    "type": "classification",
    "params": "C" ,
    "params_space": "log"
}

knn_clf = {
    "name": "knn_classification",
    "estimator": KNeighborsClassifier(),
    "type": "classification",
    "params": "n_neighbors",
    "params_space": "int"
}

knn_reg = {
    "name": "knn_regression",
    "estimator": KNeighborsRegressor(),
    "type": "regression",
    "params": "n_neighbors",
    "params_space": "int"
}

dt_reg = {
    "name": "decision_tree_regression",
    "estimator": DecisionTreeRegressor(),
    "type": "regression",
    "params": "max_depth",
    "params_space": "int"
}

dt_clf = {
    "name": "decision_tree_classification",
    "estimator": DecisionTreeClassifier(),
    "type": "classification",
    "params": "max_depth",
    "params_space": "int"
}

linear_svm = {
    "name": "linear_svm",
    "estimator": LinearSVC(),
    "type": "classification",
    "params": "C",
    "params_space": "log"
}

adaboost_clf = {
    "name": "adaboost_classification",
    "estimator": AdaBoostClassifier(),
    "type": "classification",
    "params": "learning_rate",
    "params_space": "log"
}

adaboost_reg = {
    "name": "adaboost_regression",
    "estimator": AdaBoostRegressor(),
    "type": "regression",
    "params": "learning_rate",
    "params_space": "log"
}

random_forest_clf = {
    "name": "random_forest_classification",
    "estimator": RandomForestClassifier(),
    "type": "classification",
    "params": "max_depth",
    "params_space": "int"
}

random_forest_reg = {
    "name": "random_forest_regression",
    "estimator": RandomForestRegressor(),
    "type": "regression",
    "params": "max_depth",
    "params_space": "int"
}

gaussian_nb = {
    "name": "guassian_naive_bayes",
    "estimator": GaussianNB(),
    "type": "classification"
}

models = [linear_reg, logistic_reg, knn_clf, knn_reg, linear_svm, dt_reg, dt_clf, adaboost_reg, adaboost_clf,
          random_forest_reg, random_forest_clf, gaussian_nb]
