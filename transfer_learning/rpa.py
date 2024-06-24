# Riemannian Procrustes Analysis (RPA) for transfer learning.
# This code is modified from the original code of pyriemann.
# The original code is available at https://github.com/alexandrebarachant/pyRiemann.

# Modified by: LC.Pan <panlincong@tju.edu.cn>
# Modified time: 2024-06-24 2:21:20 AM
# Modification: use raw signals instead of SPD matrices as input and output.
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
from transfer_learning.base import decode_domains
from pyriemann.utils.utils import check_weights
from pyriemann.utils.mean import mean_riemann, mean_covariance
from pyriemann.utils.distance import distance
from pyriemann.utils.base import invsqrtm, sqrtm
from pyriemann.utils.covariance import covariances
from pyriemann.transfer._rotate import _get_rotation_matrix

# recenter
class RCT(BaseEstimator, TransformerMixin): 
    """Recenter data for transfer learning.

    Recenter the data points from each domain to the Identity on manifold, ie
    make the mean of the datasets become the identity. This operation
    corresponds to a whitening step if the SPD matrices represent the spatial
    covariance matrices of multivariate signals.

    .. note::
       Using .fit() and then .transform() will give different results than
       .fit_transform(). In fact, .fit_transform() should be applied on the
       training dataset (target and source) and .transform() on the test
       partition of the target dataset.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    metric : str, default="riemann"
        Metric used for mean estimation. For the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`.
        Note, however, that only when using the "riemann" metric that we are
        ensured to re-center the data points precisely to the Identity.

    Attributes
    ----------  
    recenter_ : dict
        Dictionary containing the recentering matrices for each domain.
    
    References
    ----------
    .. [1] `Transfer Learning: A Riemannian Geometry Framework With
        Applications to Brain–Computer Interfaces
        <https://hal.archives-ouvertes.fr/hal-01923278/>`_
        P Zanini et al, IEEE Transactions on Biomedical Engineering, vol. 65,
        no. 5, pp. 1107-1116, August, 2017

    """
    def __init__(self, target_domain, metric="riemann", cov_method='lwf'): 
        self.target_domain = target_domain
        self.metric = metric
        self.cov_method = cov_method
    
    def get_recenter(self, X, sample_weight):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        C = covariances(X, estimator=self.cov_method)
        M = mean_covariance(C, self.metric, sample_weight)
        self.filters_ = invsqrtm(M)
        return self.filters_
    
    def recenter(self, X, filters_):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        return np.einsum('jk,...kl->...jl', filters_, X)
        
    def fit(self, X, y_enc, sample_weight=None):
        """Fit TLCenter.

        Calculate the mean of all matrices in each domain.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Set of raw signals.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : TLCenter instance
            The TLCenter instance.
        """
        _, _, domains = decode_domains(X, y_enc)
        n_matrices, _, _ = X.shape
        sample_weight = check_weights(sample_weight, n_matrices)
        
        self.recenter_ = {}
        for d in np.unique(domains):
            idx = domains == d
            self.recenter_[d] = self.get_recenter(
                X[idx], 
                sample_weight=sample_weight[idx]
            )
        
        return self

    def transform(self, X, y_enc=None):
        """Re-center the data points in the target domain to Identity.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Set of raw signals.
        y_enc : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_classes)
            Set of SPD matrices with mean in the Identity.
        """
        # Used during inference, apply recenter from specified target domain.
        return self.recenter(X, self.recenter_[self.target_domain])

    def fit_transform(self, X, y_enc, sample_weight=None):
        """Fit TLCenter and then transform data points.

        Calculate the mean of all matrices in each domain and then recenter
        them to Identity.

        .. note::
           This method is designed for using at training time. The output for
           .fit_transform() will be different than using .fit() and
           .transform() separately.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Set of raw signals.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_classes)
            Set of SPD matrices with mean in the Identity.
        """
        # Used during fit, in pipeline
        self.fit(X, y_enc, sample_weight)
        _, _, domains = decode_domains(X, y_enc)

        X_rct = np.zeros_like(X)
        for d in np.unique(domains):
            idx = domains == d
            X_rct[idx] = self.recenter(X[idx], self.recenter_[d])
        return X_rct

class STR(BaseEstimator, TransformerMixin):
    """Stretch data for transfer learning.

    Change the dispersion of the datapoints around their geometric mean
    for each dataset so that they all have the same desired value.

    .. note::
       Using .fit() and then .transform() will give different results than
       .fit_transform(). In fact, .fit_transform() should be applied on the
       training dataset (target and source) and .transform() on the test
       partition of the target dataset.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    dispersion : float, default=1.0
        Target value for the dispersion of the data points.
    centered_data : bool, default=False
        Whether the data has been re-centered to the Identity beforehand.
    metric : str, default="riemann"
        Metric used for calculating the dispersion.
        For the list of supported metrics,
        see :func:`pyriemann.utils.distance.distance`.

    Attributes
    ----------
    dispersions_ : dict
        Dictionary with key=domain_name and value=domain_dispersion.

    References
    ----------
    .. [1] `Riemannian Procrustes analysis: transfer learning for
        brain-computer interfaces
        <https://hal.archives-ouvertes.fr/hal-01971856>`_
        PLC Rodrigues et al, IEEE Transactions on Biomedical Engineering,
        vol. 66, no. 8, pp. 2390-2401, December, 2018

    """

    def __init__(self, target_domain, final_dispersion=1.0, centered_data=False, 
                 metric="riemann", cov_method='lwf'):
        """Init"""
        self.target_domain = target_domain
        self.final_dispersion = final_dispersion
        self.centered_data = centered_data
        self.metric = metric
        self.cov_method = cov_method

    def fit(self, X, y_enc, sample_weight=None):
        """Fit TLStretch.

        Calculate the dispersion around the mean for each domain.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Set of raw signals.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : TLStretch instance
            The TLStretch instance.
        """
        _, _, domains = decode_domains(X, y_enc)
        n_matrices, n_channels, _ = X.shape
        sample_weight = check_weights(sample_weight, n_matrices)
        covX = covariances(X, estimator=self.cov_method)
        
        self._means, self.dispersions_ = {}, {}
        for d in np.unique(domains):
            idx = domains == d
            sample_weight_d = check_weights(sample_weight[idx], np.sum(idx))
            if self.centered_data:
                self._means[d] = np.eye(n_channels)
            else:
                self._means[d] = mean_riemann(
                    covX[idx], sample_weight=sample_weight_d)

            dist = distance(
                covX[idx],
                self._means[d],
                metric=self.metric,
                squared=True,
            )
            self.dispersions_[d] = np.sum(sample_weight_d * np.squeeze(dist))

        return self

    def _center(self, X, mean):
        Mean_isqrt = invsqrtm(mean)
        return np.einsum('ij,ajt->ait', Mean_isqrt, X)

    def _uncenter(self, X, mean):
        Mean_sqrt = sqrtm(mean)
        return np.einsum('ij,ajt->ait', Mean_sqrt, X)

    def _strech(self, X, dispersion_in, dispersion_out):
        factor = np.sqrt(dispersion_out / dispersion_in)
        return X * factor

    def transform(self, X, y_enc=None):
        """Stretch the data points in the target domain.

        .. note::
           The stretching operation is properly defined only for the riemann
           metric.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Set of raw signals.
        y_enc : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Set of raw signals with desired final dispersion.
        """
        
        
        if not self.centered_data:
            # center matrices to Identity
            X = self._center(X, self._means[self.target_domain])

        # stretch
        covX = covariances(X, estimator=self.cov_method)
        X_str = self._strech(
            covX, self.dispersions_[self.target_domain], self.final_dispersion
        )

        if not self.centered_data:
            # re-center back to previous mean
            X_str = self._uncenter(X_str, self._means[self.target_domain])

        return X_str

    def fit_transform(self, X, y_enc, sample_weight=None):
        """Fit TLStretch and then transform data points.

        Calculate the dispersion around the mean for each domain and then
        stretch the data points to the desired final dispersion.

        .. note::
           This method is designed for using at training time. The output for
           .fit_transform() will be different than using .fit() and
           .transform() separately.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Set of raw signals.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Set of raw signals with desired final dispersion.
        """

        # used during fit, in pipeline
        self.fit(X, y_enc, sample_weight)
        _, _, domains = decode_domains(X, y_enc)

        X_str = np.zeros_like(X)
        for d in np.unique(domains):
            idx = domains == d

            if not self.centered_data:
                # re-center matrices to Identity
                X[idx] = self._center(X[idx], self._means[d])

            # stretch
            covX = covariances(X[idx], estimator=self.cov_method)
            X_str[idx] = self._strech(
                covX, self.dispersions_[d], self.final_dispersion
            )

            if not self.centered_data:
                # re-center back to previous mean
                X_str[idx] = self._uncenter(X_str[idx], self._means[d])

        return X_str

class ROT(BaseEstimator, TransformerMixin):
    """Rotate data for transfer learning.

    Rotate the data points from each source domain so to match its class means
    with those from the target domain. The loss function for this matching was
    first proposed in [1]_ and the optimization procedure for minimizing it
    follows the presentation from [2]_.

    .. note::
       The data points from each domain must have been re-centered to the
       identity before calculating the rotation.

    .. note::
       Using .fit() and then .transform() will give different results than
       .fit_transform(). In fact, .fit_transform() should be applied on the
       training dataset (target and source) and .transform() on the test
       partition of the target dataset.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    weights : None | array, shape (n_classes,), default=None
        Weights to assign for each class. If None, then give the same weight
        for each class.
    metric : {"euclid", "riemann"}, default="euclid"
        Metric for the distance to minimize between class means.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        the rotation matrix for each source domain in parallel. If -1 all CPUs
        are used.

    Attributes
    ----------
    rotations_ : dict
        Dictionary with key=domain_name and value=domain_rotation_matrix.

    References
    ----------
    .. [1] `Riemannian Procrustes analysis: transfer learning for
        brain-computer interfaces
        <https://hal.archives-ouvertes.fr/hal-01971856>`_
        PLC Rodrigues et al, IEEE Transactions on Biomedical Engineering,
        vol. 66, no. 8, pp. 2390-2401, December, 2018
    .. [2] `An introduction to optimization on smooth manifolds
        <https://www.nicolasboumal.net/book/>`_
        N. Boumal. To appear with Cambridge University Press. June, 2022

    """

    def __init__(self, target_domain, weights=None, metric='euclid', 
                 cov_method='lwf', n_jobs=1):
        """Init"""
        self.target_domain = target_domain
        self.cov_method = cov_method
        self.weights = weights
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y_enc, sample_weight=None):
        """Fit TLRotate.

        Calculate the rotations matrices to transform each source domain into
        the target domain.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Set of raw signals.
        y_enc : ndarray, shape (n_samples,)
            Extended labels for each sample.
        sample_weight : None | ndarray, shape (n_samples,), default=None
            Weights for each sample. If None, it uses equal weights.

        Returns
        -------
        self : TLRotate instance
            The TLRotate instance.
        """

        _, _, domains = decode_domains(X, y_enc)
        n_matrices, _, _ = X.shape
        sample_weight = check_weights(sample_weight, n_matrices)
        covX = covariances(X, estimator=self.cov_method)

        # calculate target domain means
        idx = domains == self.target_domain
        X_target, y_target = covX[idx], y_enc[idx]
        M_target = np.stack([
            mean_riemann(X_target[y_target == label],
                         sample_weight=sample_weight[idx][y_target == label])
            for label in np.unique(y_target)
        ])

        source_domains = np.unique(domains)
        source_domains = source_domains[source_domains != self.target_domain]
        rotations = Parallel(n_jobs=self.n_jobs)(
            delayed(_get_rotation_matrix)(
                np.stack([
                    mean_riemann(
                        covX[domains == d][y_enc[domains == d] == label],
                        sample_weight=sample_weight[domains == d][
                            y_enc[domains == d] == label
                        ]
                    ) for label in np.unique(y_enc[domains == d])
                ]),
                M_target,
                self.weights,
                metric=self.metric,
            ) for d in source_domains
        )

        self.rotations_ = {}
        for d, rot in zip(source_domains, rotations):
            self.rotations_[d] = rot

        return self

    def transform(self, X, y_enc=None):
        """Rotate the data points in the target domain.

        The rotations are done from source to target, so in this step the data
        points suffer no transformation at all.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Set of raw signals.
        y_enc : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Same set of raw signals as in the input.
        """

        # used during inference on target domain
        return X

    def fit_transform(self, X, y_enc, sample_weight=None):
        """Fit TLRotate and then transform data points.

        Calculate the rotation matrix for matching each source domain to the
        target domain.

        .. note::
           This method is designed for using at training time. The output for
           .fit_transform() will be different than using .fit() and
           .transform() separately.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Set of raw signals.
        y_enc : ndarray, shape (n_samples,)
            Extended labels for each sample.
        sample_weight : None | ndarray, shape (n_samples,), default=None
            Weights for each sample. If None, it uses equal weights.

        Returns
        -------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Set of raw signals after rotation step.
        """

        # used during fit in pipeline, rotate each source domain
        self.fit(X, y_enc, sample_weight)
        _, _, domains = decode_domains(X, y_enc)
        
        X_rot = np.zeros_like(X)
        
        for d in np.unique(domains):
            idx = domains == d
            if d != self.target_domain:
                X_rot[idx] = np.einsum(
                    'ij,ajt->ait',
                    self.rotations_[d],
                    X[idx]
                )
            else:
                X_rot[idx] = X[idx]
        return X_rot