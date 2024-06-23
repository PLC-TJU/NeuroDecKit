""" 
modified from metabci.brainda.algorithms.transfer_learning.mekt.py
by LC.Pan at 2024-06-23
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from metabci.brainda.algorithms.transfer_learning.mekt import mekt_kernel
from pyriemann.utils.covariance import covariances
from pyriemann.utils import mean_covariance
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.tangentspace import tangent_space
from transfer_learning import decode_domains

def mekt_feature(X, covariance_type, sample_weight=None, metric='riemann'):
    """Covariance Matrix Centroid Alignment and Tangent Space Feature Extraction.
       Parameters
    ----------
    X : ndarray
        EEG data, shape (n_trials, n_channels, n_times)

    Returns
    -------
    feature: ndarray
        feature of X, shape (n_trials, n_feature)

    """
    # Covariance Matrix Estimation
    X = covariances(X, estimator=covariance_type)
    # Covariance Matrix Centroid Alignment
    M = mean_covariance(X, metric=metric, sample_weight=sample_weight)
    iM12 = invsqrtm(M)
    C = iM12 @ X @ iM12.T
    # Tangent Space Feature Extraction
    feature = tangent_space(C, np.eye(M.shape[0]), metric=metric)

    return feature

class MEKT(BaseEstimator, TransformerMixin):
    """
    Manifold Embedded Knowledge Transfer(MEKT) [1]_.

    author: Swolf <swolfforever@gmail.com>

    Created on: 2021-01-22

    update log:
        2021-01-22 by Swolf <swolfforever@gmail.com>

        2023-12-09 by heoohuan <heoohuan@163.com>（Add code annotation）
        
        2024-06-23 by LC.Pan <coreylin@outlook.com>
         - add transform method 
         - add target_domain parameter to fit_transform method
         - add decode_domains function to decode source and target domains
         - add sample_weight parameter to mekt_feature method
         - add metric parameter to mekt_feature method

    Parameters
    ----------
    subspace_dim: int
        Selected projection vector, by default 10.
    max_iter: int
        max iterations, by default 5.
    alpha: float
        regularized term for source domain discriminability, by default 0.01.
    beta: float
        regularized term for target domain locality, by default 0.1.
    rho: float
        regularized term for parameter transfer, by default 20.
    k: int
        number of nearest neighbors.
    t: int
        heat kernel parameter.
    covariance_type: str
        Covariance category, by default 'lwf'.

    Attributes
    ----------
    subspace_dim: int
        Selected projection vector, by default 10.
    max_iter: int
        max iterations, by default 5.
    alpha: float
        regularized term for source domain discriminability, by default 0.01.
    beta: float
        regularized term for target domain locality, by default 0.1.
    rho: float
        regularized term for parameter transfer, by default 20.
    k: int
        number of nearest neighbors.
    t: int
        heat kernel parameter.
    covariance_type: str
        covariance category, by default 'lwf'.
    A_: ndarray
        first type center, shape(n_class, n_channels, n_channels).
    B_: ndarray
       second type center, shape(n_class, n_channels, n_channels).

    Raises
    ----------
    ValueError
        None

    References
    ----------
    .. [1] Zhang W, Wu D. Manifold embedded knowledge transfer for brain-computer interfaces
       [J].IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2020, 28 (5): 1117–1127.

    """

    def __init__(
        self,
        target_domain,
        subspace_dim: int = 10,
        max_iter: int = 5,
        alpha: float = 0.01,
        beta: float = 0.1,
        rho: float = 20,
        k: int = 10,
        t: int = 1,
        covariance_type="lwf",
        metric="riemann",
    ):
        self.target_domain = target_domain
        self.subspace_dim = subspace_dim
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.k = k
        self.t = t
        self.covariance_type = covariance_type
        self.metric = metric
        self.A_ = None
        self.B_ = None
    
    def transform(self, X):
        """Obtain target domain features after MEKT transformation.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_times).

        Returns
        -------
        target_features: ndarray
            target domain features, shape(n_trials, n_features).

        """
        feature = mekt_feature(X, self.covariance_type, metric=self.metric)
        target_features = feature @ self.B_
        return target_features

    def fit_transform(self, X, y_enc, sample_weight=None):
        """Obtain source and target domain features after MEKT transformation.

        Parameters
        ----------
        X: ndarray
            EEG data, shape(n_trials, n_channels, n_times).
        y_enc: ndarray
            Label, shape(n_trials,).
        sample_weight: ndarray
            Sample weight, shape(n_trials,).

        Returns
        -------
        feature: ndarray
            source and target domain features, shape(n_trials, n_features).

        """
        X, y, domains = decode_domains(X, y_enc)
        
        Xs = X[domains != self.target_domain]
        ys = y[domains != self.target_domain]
        Xt = X[domains == self.target_domain]
        
        featureXs = mekt_feature(
            Xs, 
            self.covariance_type, 
            sample_weight=sample_weight, 
            metric=self.metric
            )
        featureXt = mekt_feature(
            Xt, 
            self.covariance_type, 
            metric=self.metric
            )
        self.A_, self.B_ = mekt_kernel(
            featureXs,
            featureXt,
            ys,
            d=self.subspace_dim,
            max_iter=self.max_iter,
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            k=self.k,
            t=self.t,
        )
        source_features = featureXs @ self.A_
        target_features = featureXt @ self.B_
        feature = np.concatenate((source_features, target_features), axis=0)
        return feature