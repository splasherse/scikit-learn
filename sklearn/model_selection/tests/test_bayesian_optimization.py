import pytest
from sklearn.model_selection import BayesianOptimizationSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import SVC


def test_bayesian_optimization_search():
    X, y = load_iris(return_X_y=True)
    param_distributions = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1]
    }
    model = SVC()
    search = BayesianOptimizationSearchCV(model, param_distributions, n_iter=10, cv=3)
    search.fit(X, y)
    assert search.best_score_ > 0.5
