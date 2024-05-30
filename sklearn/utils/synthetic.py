import numpy as np
from sklearn.utils import check_array

def generate_synthetic_data(X, method='random', n_samples=100):
    """Generate synthetic data based on the input dataset X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.

    method : str, optional (default='random')
        The method to generate synthetic data. Options are 'random', 'noise', or 'interpolation'.

    n_samples : int, optional (default=100)
        The number of synthetic samples to generate.

    Returns
    -------
    X_synthetic : array-like of shape (n_samples, n_features)
        The generated synthetic data.
    """
    X = check_array(X)

    if method not in ['random', 'noise', 'interpolation']:
        raise ValueError("Method must be one of 'random', 'noise', or 'interpolation'")

    n_features = X.shape[1]

    if method == 'random':
        X_synthetic = np.random.rand(n_samples, n_features) * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)

    elif method == 'noise':
        noise = np.random.normal(0, 0.1, size=(n_samples, n_features))
        X_synthetic = X[np.random.choice(X.shape[0], n_samples)] + noise

    elif method == 'interpolation':
        indices = np.random.choice(X.shape[0], n_samples)
        X_synthetic = (X[indices] + X[np.random.choice(X.shape[0], n_samples)]) / 2

    return X_synthetic
