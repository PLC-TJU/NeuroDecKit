

from sklearn.pipeline import Pipeline
from inspect import signature

def chesk_sample_weight(clf):
    if isinstance(clf, Pipeline):
        return chesk_sample_weight(clf.steps[-1][1])
    else:
        fit_method = getattr(clf, 'fit')
        params = signature(fit_method).parameters
        return 'sample_weight' in params

from pyriemann.utils.geodesic import geodesic
from pyriemann.utils.base import invsqrtm
def recursive_reference_center(reference_old, X_new, alpha, metric='riemann'):
    """Recursive reference centering.

    Parameters
    ----------
    reference_old : ndarray, shape (n_channels, n_channels)
        The reference matrix to be updated.
    X_new : ndarray, shape (1, n_channels, n_channels) or (n_channels, n_channels)
        The new matrices to be centered.
    alpha : float
        The weight to assign to the new samples.
    metric : str, default="riemann"
        The metric to use for the geodesic distance.

    Returns
    -------
    reference_new : ndarray, shape (n_channels, n_channels)
        The updated reference matrix. 
    """
    X_new = X_new.copy() 
    reference_old = reference_old.copy()
    X_new = X_new.reshape((-1, *X_new.shape[-2:]))
    X_new = X_new.mean(axis=0, keepdims=False)
    C = geodesic(reference_old, X_new, alpha, metric=metric)
    reference_new = invsqrtm(C)
    reference_new = reference_new.reshape(reference_old.shape)
    return reference_new
    
  