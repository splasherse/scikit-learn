import pytest
import numpy as np
from sklearn.utils._synthetic import generate_synthetic_data

def test_generate_synthetic_data_random():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_synthetic = generate_synthetic_data(X, method='random', n_samples=5)
    assert X_synthetic.shape == (5, 2)
    assert np.all(X_synthetic >= X.min(axis=0)) and np.all(X_synthetic <= X.max(axis=0))

def test_generate_synthetic_data_noise():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_synthetic = generate_synthetic_data(X, method='noise', n_samples=5)
    assert X_synthetic.shape == (5, 2)

def test_generate_synthetic_data_interpolation():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_synthetic = generate_synthetic_data(X, method='interpolation', n_samples=5)
    assert X_synthetic.shape == (5, 2)

def test_generate_synthetic_data_invalid_method():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        generate_synthetic_data(X, method='invalid_method', n_samples=5)
