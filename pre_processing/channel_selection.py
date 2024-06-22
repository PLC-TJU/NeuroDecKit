""" 
This is a modified version of the original code from pyriemann.channelselection.
 -Add: CSPChannelSelector

Author: Pan.LC <coreylin2023@outlook.com>
Date: 2024/6/21
License: MIT License
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.distance import distance
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from scipy.linalg import eigh
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.ajd import ajd_pham

class RiemannChannelSelector(BaseEstimator, TransformerMixin):
    """Channel selection based on a Riemannian geometry criterion.
    
    Parameters
    ----------
    nelec : int, default=16
        The number of electrode to keep in the final subset.
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    covmeans_ : ndarray, shape (n_classes, n_channels, n_channels)
        Centroids for each class.
    dist_ : list
        Distance at each iteration.
    subelec_ : list
        Indices of selected channels.
    """

    def __init__(self, nelec=16, metric="riemann", n_jobs=1):
        """Init."""
        self.nelec = nelec
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None):
        """Find the optimal subset of electrodes.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Set of raw EEG signals.
        y : None | ndarray, shape (n_samples,), default=None
            Labels for each signal.
        sample_weight : None | ndarray, shape (n_samples,), default=None
            Weights for each signal. If None, it uses equal weights.

        Returns
        -------
        self : ElectrodeSelection instance
            The ElectrodeSelection instance.
        """
        # Compute covariance matrices
        Cov = Covariances(estimator='lwf').transform(X)

        if y is None:
            y = np.ones((X.shape[0]))

        mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)
        mdm.fit(Cov, y, sample_weight=sample_weight)
        self.covmeans_ = mdm.covmeans_

        n_classes, n_channels, _ = self.covmeans_.shape

        self.dist_ = []
        self.subelec_ = list(range(n_channels))
        while len(self.subelec_) > self.nelec:
            di = np.zeros((len(self.subelec_), 1))
            for idx in range(len(self.subelec_)):
                sub = self.subelec_[:]
                sub.pop(idx)
                di[idx] = 0
                for i in range(n_classes):
                    for j in range(i + 1, n_classes):
                        di[idx] += distance(
                            self.covmeans_[i][:, sub][sub, :],
                            self.covmeans_[j][:, sub][sub, :],
                            metric=mdm.metric_dist,
                        )

            torm = di.argmax()
            self.dist_.append(di.max())
            self.subelec_.pop(torm)
        return self

    def transform(self, X):
        """Return reduced signals.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Set of raw EEG signals.

        Returns
        -------
        X_new : ndarray, shape (n_samples, n_elec, n_times)
            Set of EEG signals after reduction of the number of channels.
        """
        return X[:, self.subelec_, :]


class CSPChannelSelector(BaseEstimator, TransformerMixin):
    def __init__(self, nelec=16, metric='euclid'):
        """Init."""
        self.nelec = nelec
        self.metric = metric

    def fit(self, X, y):
        """Train CSP spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Set of covariance matrices.
        y : ndarray, shape (n_trials,)
            Labels for each trial.

        Returns
        -------
        self : CSP instance
            The CSP instance.
        """
        if not isinstance(self.nfilter, int):
            raise TypeError('nfilter must be an integer')
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError('X must be an array.')
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError('y must be an array.')
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 3:
            raise ValueError('X must be n_trials * n_channels * n_channels')
        if len(y) != len(X):
            raise ValueError('X and y must have the same length.')
        if np.squeeze(y).ndim != 1:
            raise ValueError('y must be of shape (n_trials,).')

        n_trials, n_channels, _ = X.shape
        classes = np.unique(y)
        
        # Compute covariance matrices
        Cov = Covariances(estimator='lwf').transform(X)
        
        # estimate class means
        C = []
        for c in classes:
            C.append(mean_covariance(Cov[y == c], self.metric))
        C = np.array(C)

        # Switch between binary and multiclass
        if len(classes) == 2:
            evals, evecs = eigh(C[1], C[0] + C[1])
            # sort eigenvectors
            ix = np.argsort(np.abs(evals - 0.5))[::-1]
        elif len(classes) > 2:
            evecs, D = ajd_pham(C)
            Ctot = mean_covariance(C, self.metric)
            evecs = evecs.T

            # normalize
            for i in range(evecs.shape[1]):
                tmp = evecs[:, i].T @ Ctot @ evecs[:, i]
                evecs[:, i] /= np.sqrt(tmp)

            mutual_info = []
            # class probability
            Pc = [np.mean(y == c) for c in classes]
            for j in range(evecs.shape[1]):
                a = 0
                b = 0
                for i, c in enumerate(classes):
                    tmp = evecs[:, j].T @ C[i] @ evecs[:, j]
                    a += Pc[i] * np.log(np.sqrt(tmp))
                    b += Pc[i] * (tmp ** 2 - 1)
                mi = - (a + (3.0 / 16) * (b ** 2))
                mutual_info.append(mi)
            ix = np.argsort(mutual_info)[::-1]
        else:
            raise ValueError("Number of classes must be >= 2.")

        self.subelec_ = ix[:self.nelec]

        return self

    def transform(self, X):
        """Return reduced signals.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Set of raw EEG signals.

        Returns
        -------
        X_new : ndarray, shape (n_samples, n_elec, n_times)
            Set of EEG signals after reduction of the number of channels.
        """
        return X[:, self.subelec_, :]
    