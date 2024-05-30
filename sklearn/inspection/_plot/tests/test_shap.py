import pytest
import shap
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection._shap import shap_values, plot_shap_summary


def test_shap_values():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier().fit(X, y)
    shap_vals = shap_values(model, X)
    assert shap_vals.shape[0] == X.shape[0]


def test_plot_shap_summary():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier().fit(X, y)
    plot_shap_summary(model, X)
