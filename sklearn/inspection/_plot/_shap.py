import shap

def shap_values(model, X):
    """Compute SHAP values for a given model and dataset.

    Parameters
    ----------
    model : estimator object
        The scikit-learn estimator.

    X : array-like of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    shap_values : array
        SHAP values for the input samples.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap_values

def plot_shap_summary(model, X):
    """Plot a summary of SHAP values for a given model and dataset.

    Parameters
    ----------
    model : estimator object
        The scikit-learn estimator.

    X : array-like of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    None
    """
    shap_vals = shap_values(model, X)
    shap.summary_plot(shap_vals, X)
