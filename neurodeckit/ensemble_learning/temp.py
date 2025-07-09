from ..transfer_learning import decode_domains

from sklearn.ensemble._stacking import _BaseStacking

from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np

from sklearn.base import (
    clone,
    is_classifier,
)

from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    _raise_for_unsupported_routing,
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    check_is_fitted,
)

from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.metadata_routing import _routing_enabled


def _fit_single_estimator(
    estimator, X, y, fit_params, message_clsname=None, message=None
):
    """Private function used to fit an estimator within a job."""
    # TODO(SLEP6): remove if condition for unrouted sample_weight when metadata
    # routing can't be disabled.
    if not _routing_enabled() and "sample_weight" in fit_params:
        try:
            with _print_elapsed_time(message_clsname, message):
                estimator.fit(X, y, sample_weight=fit_params["sample_weight"])
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError(
                    "Underlying estimator {} does not support sample weights.".format(
                        estimator.__class__.__name__
                    )
                ) from exc
            raise
    else:
        with _print_elapsed_time(message_clsname, message):
            estimator.fit(X, y, **fit_params)
    return estimator

class _BaseDomainAdaptiveStacking(_BaseStacking):
    """Base class for domain-adaptive stacking estimators."""
    
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        verbose=0,
        passthrough=False,
        domain_group_strategy='leave_one_domain_out'
    ):
        super().__init__(estimators=estimators)
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.passthrough = passthrough
        self.domain_group_strategy = domain_group_strategy
        
    def _decode_domains(self, X, y_enc):
        """Decode domain information from encoded labels."""
        _, y, domain_tags = decode_domains(X, y_enc)
        return y, domain_tags
        
    def _create_domain_groups(self, domain_tags):
        """Create domain-based groups for cross-validation."""
        unique_domains = np.unique(domain_tags)
        domain_to_group = {domain: idx for idx, domain in enumerate(unique_domains)}
        return np.array([domain_to_group[d] for d in domain_tags])
    
    def _create_domain_aware_cv(self, X, y, domain_tags):
        """Create domain-aware cross-validation splits."""
        groups = self._create_domain_groups(domain_tags)
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
        
        # If user provided custom CV, use it directly
        if self.cv != "prefit" and hasattr(cv, 'split'):
            return cv.split(X, y, groups=groups)
            
        # Create domain-aware splits based on strategy
        unique_groups = np.unique(groups)
        splits = []
        
        if self.domain_group_strategy == 'leave_one_domain_out':
            for test_group in unique_groups:
                test_mask = groups == test_group
                train_mask = ~test_mask
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                splits.append((train_idx, test_idx))
                
        elif self.domain_group_strategy == 'domain_kfold':
            from sklearn.model_selection import GroupKFold
            return GroupKFold(n_splits=self.cv if isinstance(self.cv, int) else 5
                ).split(X, y, groups=groups)
        
        return splits
    
    def _fit_base_estimators(self, X, y_enc, sample_weight=None):
        """Fit base estimators with domain-encoded labels."""
        # all_estimators contains all estimators, the one to be fitted and the
        # 'drop' string.
        names, all_estimators = self._validate_estimators()
        
        # FIXME: when adding support for metadata routing in Stacking*.
        # This is a hotfix to make StackingClassifier and StackingRegressor
        # pass the tests despite not supporting metadata routing but sharing
        # the same base class with VotingClassifier and VotingRegressor.
        self.fit_params = dict()
        if sample_weight is not None:
            self.fit_params["sample_weight"] = sample_weight

        stack_method = [self.stack_method] * len(all_estimators)

        if self.cv == "prefit":
            self.estimators_ = []
            for estimator in all_estimators:
                if estimator != "drop":
                    check_is_fitted(estimator)
                    self.estimators_.append(estimator)
        else:
            # Fit the base estimators on the whole training data. Those
            # base estimators will be used in transform, predict, and
            # predict_proba. They are exposed publicly.
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single_estimator)(clone(est), X, y_enc, self.fit_params)
                for est in all_estimators
                if est != "drop"
            )

        self.named_estimators_ = Bunch()
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            if org_est != "drop":
                current_estimator = self.estimators_[est_fitted_idx]
                self.named_estimators_[name_est] = current_estimator
                est_fitted_idx += 1
                if hasattr(current_estimator, "feature_names_in_"):
                    self.feature_names_in_ = current_estimator.feature_names_in_
            else:
                self.named_estimators_[name_est] = "drop"

        self.stack_method_ = [
            self._method_name(name, est, meth)
            for name, est, meth in zip(names, all_estimators, stack_method)
        ]
        
    def _generate_meta_features(self, X, y_enc, y_original, domain_tags):
        """Generate meta-features using domain-aware cross-validation."""
        # Get domain-aware CV splits
        cv_splits = self._create_domain_aware_cv(X, y_original, domain_tags)
        
        predictions = []
        for estimator, method in zip(self.estimators_, self.stack_method_):
            if estimator == "drop":
                continue
                
            pred = cross_val_predict(
                estimator, X, y_enc, cv=cv_splits, method=method, 
                n_jobs=self.n_jobs, verbose=self.verbose
            )
            predictions.append(pred)
            
        return self._concatenate_predictions(X, predictions)
    
    def _fit_final_estimator(self, X_meta, y_original):
        """Fit final estimator with original labels."""
        if self.final_estimator is None:
            return
        
        # Fit final estimator with original labels
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(X_meta, y_original)
    
    def fit(self, X, y_enc, sample_weight=None):
        """Fit the domain-adaptive stacking model.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors.
            
        y_enc : array-like of shape (n_samples,)
            Domain-encoded target values.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        self : object
        """
        # Decode domain information
        y_original, domain_tags = self._decode_domains(X, y_enc)
        self.domain_tags_ = domain_tags
        
        # Validate and process inputs
        X, y_enc = check_X_y(X, y_enc, accept_sparse=True, allow_nd=True)
        self._validate_final_estimator()
        
        # Fit base estimators with domain-encoded labels
        self._fit_base_estimators(X, y_enc, sample_weight)
        
        # Generate meta-features using domain-aware CV
        if self.cv == "prefit":
            # For prefit, use direct predictions
            predictions = [
                getattr(est, meth)(X)
                for est, meth in zip(self.estimators_, self.stack_method_)
                if est != "drop"
            ]
            X_meta = self._concatenate_predictions(X, predictions)
        else:
            X_meta = self._generate_meta_features(X, y_enc, y_original, domain_tags)
        
        # Only not None or not 'drop' estimators will be used in transform
        self.stack_method_ = [
            meth
            for meth, est in zip(self.stack_method_, self.estimators_)
            if est != "drop"
        ]
        
        # Fit final estimator with original labels
        self._fit_final_estimator(X_meta, y_original)
        
        return self
    

class DomainAdaptiveStackingClassifier(_BaseDomainAdaptiveStacking, StackingClassifier):
    
    def fit(self, X, y_enc, sample_weight=None):
        """Fit the domain-adaptive stacking classifier.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors.
            
        y_enc : array-like of shape (n_samples,)
            Domain-encoded target values.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        self : object
        """
        _raise_for_unsupported_routing(self, "fit", sample_weight=sample_weight)
        
        # Decode domains and get original labels
        y_original, domain_tags = self._decode_domains(X, y_enc)
        
        # Validate classification targets
        check_classification_targets(y_original)
        self.classes_ = np.unique(y_original)
        
        # Fit using base class implementation
        super().fit(X, y_enc, sample_weight)
        return self
    
    def predict(self, X, **predict_params):

        check_is_fitted(self)
        X_meta = self.transform(X)
        return self.final_estimator_.predict(X_meta, **predict_params)
    
    def predict_proba(self, X):

        check_is_fitted(self)
        X_meta = self.transform(X)
        return self.final_estimator_.predict_proba(X_meta)
    
    def decision_function(self, X):

        check_is_fitted(self)
        X_meta = self.transform(X)
        return self.final_estimator_.decision_function(X_meta)
    
    def get_domain_importances(self):
        """Get domain importance weights from final estimator (if available)."""
        if hasattr(self.final_estimator_, 'coef_'):
            return self.final_estimator_.coef_
        elif hasattr(self.final_estimator_, 'feature_importances_'):
            return self.final_estimator_.feature_importances_
        return None
    
    def get_domain_predictions(self, X):
        """Get base model predictions for each domain."""
        domain_preds = {}
        for name, estimator in self.named_estimators_.items():
            if estimator == "drop":
                continue
                
            # Decode domain from predictions
            preds = estimator.predict(X)
            _, _, domains = decode_domains(X, preds)
            
            for domain in np.unique(domains):
                if domain not in domain_preds:
                    domain_preds[domain] = {}
                domain_mask = domains == domain
                domain_preds[domain][name] = preds[domain_mask]
                
        return domain_preds