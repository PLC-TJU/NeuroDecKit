"""
This module implements domain-adaptive stacking for multi-domain classification.

The module is based on the scikit-learn implementation of stacking, and supports the following features:

- Domain-adaptive stacking: The module supports domain-adaptive stacking, which means that the target domain 
    is used to train the base estimators and the final estimator, while the other domains are used to train 
    the meta-features.
- Multi-domain classification: The module supports multi-domain classification, which means that the base 
    estimators can be trained on multiple domains, and the final estimator can be trained on the target domain.
- Group-based cross-validation: The module supports group-based cross-validation, which means that the base 
    estimators are trained on different groups of domains, and the meta-features are trained on the same 
    groups of domains.
- Sample weights: The module supports sample weights for each domain, which means that the base estimators are 
    trained on different subsets of the data, and the meta-features are trained on the same subsets of the data.

The module is designed to be used as a drop-in replacement for scikit-learn's stacking implementation. 
The only difference is that the `fit` method of the `DomainAdaptiveStackingClassifier` class takes an 
additional `target_domain` parameter, which specifies the target domain for multi-domain classification.

Authors: LC, Pan <panlincong@tju.edu.cn>
Date: 2025-07-06
"""

import numpy as np
from copy import deepcopy
from numbers import Integral
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble._stacking import _BaseStacking
from sklearn.base import (
    clone,
    is_classifier,
    ClassifierMixin,
)
from sklearn.model_selection import check_cv, GroupKFold
from sklearn.utils.metadata_routing import (
    _raise_for_unsupported_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y
)
from sklearn.utils._param_validation import (
    HasMethods,
    StrOptions,
    validate_params,
)
from sklearn.utils import indexable
from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.metadata_routing import _routing_enabled
from ..transfer_learning import decode_domains


@validate_params(
    {
        "estimator": [HasMethods(["fit", "predict"])],
        "X": ["array-like", "sparse matrix"],
        "y_enc": ["array-like",  None],
        "cv_splits": [list, tuple],
        "n_jobs": [Integral, None],
        "fit_params": [dict, None],
        "method": [
            StrOptions(
                {
                    "predict",
                    "predict_proba",
                    "predict_log_proba",
                    "decision_function",
                }
            )
        ],
    },
    prefer_skip_nested_validation=False # 跳过嵌套验证
)
def _cross_val_predict(estimator, X, y_enc, cv_splits, method="predict", fit_params=None, n_jobs=None):
    """自定义交叉验证预测函数，正确处理域编码标签"""
    
    # 验证并准备输入数据
    X, y_enc = indexable(X, y_enc)
    n_samples = X.shape[0]
    
    # 验证预测方法
    if not hasattr(estimator, method):
        raise ValueError(f"Estimator does not implement the '{method}' method")
    
    # 验证交叉验证分割
    if len(cv_splits) == 0:
        raise ValueError("No cross-validation splits provided")    
    
    # 检查所有测试索引是否覆盖所有样本(确保可用于meta-features)
    all_test_indices = np.concatenate([test_idx for _, test_idx in cv_splits])
    if len(np.unique(all_test_indices)) != n_samples:
        raise ValueError("Cross-validation splits do not cover all samples")
    
    # 确定预测结果的形状和数据类型
    try:
        # 使用样本子集安全测试预测形状
        test_idx_sample = cv_splits[0][1][:1]  # 取第一个测试集的第一个样本
        sample_pred = getattr(deepcopy(estimator), method)(X[test_idx_sample])
        
        if sample_pred.ndim == 1:
            predictions = np.empty(n_samples, dtype=sample_pred.dtype)
            output_shape = "1d"
        else:
            predictions = np.empty((n_samples, sample_pred.shape[1]), dtype=sample_pred.dtype)
            output_shape = "2d"
    except Exception as e:
        raise RuntimeError("Failed to determine prediction shape") from e
    
    # 定义每个fold的处理函数
    def _fit_predict_fold(train_idx, test_idx):
        """处理单个fold的训练和预测"""
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y_enc[train_idx] if y_enc is not None else None
        
        cloned_estimator = deepcopy(estimator)
        
        # 处理fit参数
        fold_fit_params = deepcopy(fit_params) if fit_params else {}
        
        # 添加样本权重支持
        if "sample_weight" in fold_fit_params:
            fold_fit_params["sample_weight"] = fold_fit_params["sample_weight"][train_idx]
        
        # 训练模型
        if y_train is not None:
            cloned_estimator.fit(X_train, y_train, **fold_fit_params)
        else:
            cloned_estimator.fit(X_train, **fold_fit_params)
        
        # 进行预测
        pred = getattr(cloned_estimator, method)(X_test)
        
        # 处理预测形状
        if output_shape == "1d" and pred.ndim > 1:
            # 对于预期1D但返回2D的情况，取第一列
            if pred.shape[1] == 1:
                pred = pred.ravel()
            else:
                raise ValueError("Prediction shape mismatch")
        elif output_shape == "2d" and pred.ndim == 1:
            # 对于预期2D但返回1D的情况，转换为2D
            pred = pred.reshape(-1, 1)
        
        return test_idx, pred
    
    # 并行处理所有fold
    try:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_predict_fold)(train_idx, test_idx)
            for train_idx, test_idx in cv_splits
        )
    except Exception as e:
        raise RuntimeError("Parallel execution failed") from e
    
    # 合并结果
    for test_idx, pred in results:
        # 验证索引范围
        if np.max(test_idx) >= n_samples or np.min(test_idx) < 0:
            raise ValueError(f"Invalid test indices: min={np.min(test_idx)}, max={np.max(test_idx)}")
        
        # 验证预测形状
        if output_shape == "1d":
            if pred.ndim != 1:
                raise ValueError(f"Expected 1D predictions, got {pred.ndim}D")
            if len(pred) != len(test_idx):
                raise ValueError("Prediction length does not match test indices")
            predictions[test_idx] = pred
        else:
            if pred.ndim != 2:
                raise ValueError(f"Expected 2D predictions, got {pred.ndim}D")
            if pred.shape[0] != len(test_idx):
                raise ValueError("Prediction rows do not match test indices")
            predictions[test_idx, :] = pred
    
    return predictions


def _fit_single_estimator(
    estimator, X, y, fit_params, message_clsname=None, message=None
):
    """Private function used to fit an estimator within a job."""
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

def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.

    First, we check the fitted `final_estimator_` if available, otherwise we check the
    unfitted `final_estimator`. We raise the original `AttributeError` if `attr` does
    not exist. This function is used together with `available_if`.
    """

    def check(self):
        if hasattr(self, "final_estimator_"):
            getattr(self.final_estimator_, attr)
        else:
            getattr(self.final_estimator, attr)

        return True

    return check

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
        target_domain=None, # important parameter
        domain_group_strategy='leave_one_domain_out',
    ):
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method=stack_method,
            n_jobs=n_jobs,
            verbose=verbose,
            passthrough=passthrough
        )
        self.target_domain = target_domain
        self.domain_group_strategy = domain_group_strategy
        self.fit_params = None
        self.estimators_ = None
        self.named_estimators_ = None
        self.stack_method_ = None
        self.feature_names_in_ = None
        self.final_estimator_ = None
        
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
        """创建域感知交叉验证分割，支持target_domain处理"""
        unique_domains = np.unique(domain_tags)
        if len(unique_domains) == 1:
            # 单域情况：使用标准的K折交叉验证
            return list(check_cv(self.cv, y, classifier=is_classifier(self)).split(X, y))
    
        # 如果有指定目标域且目标域存在
        if self.target_domain and self.target_domain in domain_tags:
            # 创建单次分割：非目标域作为训练集，全部作为测试集  
            train_idx = np.where(domain_tags != self.target_domain)[0]
            test_idx = np.arange(len(domain_tags))
            return [(train_idx, test_idx)]
        
        # 处理多域无目标域情况
        groups = self._create_domain_groups(domain_tags)
        
        # 使用check_cv验证和完善交叉验证策略
        cv = check_cv(self.cv, y, classifier=is_classifier(self))
        
        # 如果有自定义CV且支持分组，直接使用
        if hasattr(cv, 'split') and hasattr(cv, 'groups'):
            return list(cv.split(X, y, groups=groups))
            
        # 处理特定域分组策略
        unique_groups = np.unique(groups)
        splits = []
        
        if self.domain_group_strategy == 'leave_one_domain_out':
            # 对于某些有监督方法，leave_one_domain_out策略并不适用（例如MEKT），
            for test_group in unique_groups:
                test_mask = groups == test_group
                train_mask = ~test_mask
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                splits.append((train_idx, test_idx))
                
        elif self.domain_group_strategy == 'domain_kfold':
            n_splits = min(self.cv if isinstance(self.cv, int) else 5, 
                           len(unique_domains))
            cv = GroupKFold(n_splits=n_splits)
            return list(cv.split(X, y, groups=groups))
        
        else:
            # 5折交叉验证
            cv = check_cv(5, y, classifier=is_classifier(self))
            for train_idx, test_idx in cv.split(X, y, groups=groups):
                splits.append((train_idx, test_idx))
        
        return splits
    
    def _fit_base_estimators(self, X, y_enc, sample_weight=None):
        """Fit base estimators with domain-encoded labels."""
        # all_estimators contains all estimators, the one to be fitted and the
        # 'drop' string.
        names, all_estimators = self._validate_estimators()
        
        # Prepare fit parameters
        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        stack_method = [self.stack_method] * len(all_estimators)

        if self.cv == "prefit":
            self.estimators_ = []
            for estimator in all_estimators:
                if estimator != "drop":
                    check_is_fitted(estimator)
                    self.estimators_.append(estimator)
                
        else:
            
            # Fit the base estimators on the whole training data
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single_estimator)(deepcopy(est), X, y_enc, fit_params)
                for est in all_estimators
                if est != "drop"
            )

        self.named_estimators_ = {}
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
        
        # For prefit, use direct predictions
        if self.cv == "prefit" or cv_splits is None:
            predictions = [
                getattr(est, meth)(X)
                for est, meth in zip(self.estimators_, self.stack_method_)
                if est != "drop"
            ]
            return self._concatenate_predictions(X, predictions)
        
        # Generate meta-features using domain-aware CV
        predictions = []
        for estimator, method in zip(self.estimators_, self.stack_method_):
            if estimator == "drop":
                continue
                
            pred = _cross_val_predict(
                estimator, X, y_enc, cv_splits=deepcopy(cv_splits), method=method, 
                fit_params=getattr(self, "fit_params", {}), n_jobs=self.n_jobs
            )

            predictions.append(pred)
            
        return self._concatenate_predictions(X, predictions)
    
    def _fit_final_estimator(self, X_meta, y_original):
        """Fit final estimator with original labels."""
        if X_meta.shape[0] != len(y_original):
            raise ValueError("X_meta and y_original have inconsistent numbers of samples")
        
        if self.final_estimator is None:
            # Default to LogisticRegression if no final estimator provided
            self.final_estimator = LogisticRegression()
            
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(X_meta, y_original)
    
    def fit(self, X, y_enc, sample_weight=None):
        """Fit the domain-adaptive stacking model."""
        # Decode domain information
        y_original, domain_tags = self._decode_domains(X, y_enc)
        self.domain_tags_ = domain_tags
        self.y_original_ = y_original
        self.classes_ = np.unique(y_original)
        
        # Validate and process inputs
        X, y_enc = check_X_y(X, y_enc, accept_sparse=True, allow_nd=True)
        self._validate_final_estimator()
        
        # Check if target domain is valid
        if self.target_domain and self.target_domain not in domain_tags:
            self.target_domain = None
        
        # Fit base estimators with domain-encoded labels
        self._fit_base_estimators(X, y_enc, sample_weight)
        
        # Generate meta-features using domain-aware CV
        X_meta = self._generate_meta_features(X, y_enc, y_original, domain_tags)
        
        # Filter out dropped estimators
        self.stack_method_ = [
            meth
            for meth, est in zip(self.stack_method_, self.estimators_)
            if est != "drop"
        ]
        
        # Fit final estimator with original labels
        self._fit_final_estimator(X_meta, y_original)
        
        return self
    

class DomainAdaptiveStackingClassifier(_BaseDomainAdaptiveStacking, ClassifierMixin):
    """Stacking classifier for domain adaptation scenarios with encoded domain labels."""
    _parameter_constraints: dict = {
        **_BaseDomainAdaptiveStacking._parameter_constraints,
        "stack_method": [
            StrOptions({"auto", "predict_proba", "decision_function", "predict"})
        ],
    }
    
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
        target_domain=None,
        domain_group_strategy='leave_one_domain_out',
    ):
        """
        Stacking classifier for domain adaptation with target domain support.
        
        Parameters:
        -----------
        estimators : list of (str, estimator)
            Base estimators to stack.
            
        final_estimator : estimator, optional
            Meta-classifier to combine base predictions.
            
        cv : int, cross-validator or "prefit", optional
            Cross-validation strategy.
            
        stack_method : {'auto', 'predict_proba', 'decision_function', 'predict'}, default='auto'
            Prediction method for base estimators.
            
        n_jobs : int, optional
            Number of parallel jobs.
            
        verbose : int, default=0
            Verbosity level.
            
        passthrough : bool, default=False
            Whether to include original features in meta-features.
        
        target_domain : str, optional
            Specific domain to treat as target (test) domain during training.
            
        domain_group_strategy : {'leave_one_domain_out', 'domain_kfold'}, default='leave_one_domain_out'
            Strategy for grouping domains during CV.
            
        """
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method=stack_method,
            n_jobs=n_jobs,
            verbose=verbose,
            passthrough=passthrough,
            target_domain=target_domain,
            domain_group_strategy=domain_group_strategy,
        )
        self.classes_ = None
        self.y_original_ = None
        self.domain_tags_ = None
        self.feature_names_in_ = None
    
    def _validate_final_estimator(self):
        """Ensure the final estimator is a classifier."""
        if self.final_estimator is None:
            self.final_estimator = LogisticRegression()
            
        self.final_estimator_ = clone(self.final_estimator)
        if not is_classifier(self.final_estimator_):
            raise ValueError(
                "'final_estimator' should be a classifier. Got {}".format(
                    self.final_estimator_.__class__.__name__
                )
            )
    
    def fit(self, X, y_enc, sample_weight=None):
        """Fit the domain-adaptive stacking classifier."""
        # Validate metadata routing
        _raise_for_unsupported_routing(self, "fit", sample_weight=sample_weight)
        
        # Print verbose message
        if self.verbose > 0:
            print(f"Starting fit with {len(self.estimators)} base estimators")
            if self.target_domain:
                print(f"Using target_domain: {self.target_domain}")
        
        # Call parent fit method
        super().fit(X, y_enc, sample_weight)
        
        # Set classes for classification
        self.classes_ = np.unique(self.y_original_)
        
        if self.verbose > 0:
            print("Fit completed successfully")
            print(f"Final estimator: {type(self.final_estimator_).__name__}")
        
        return self
    
    @available_if(_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Predict class labels for X."""
        check_is_fitted(self)
        X_meta = self.transform(X)
        return self.final_estimator_.predict(X_meta, **predict_params)
    
    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Predict class probabilities for X."""
        check_is_fitted(self)
        X_meta = self.transform(X)
        proba = self.final_estimator_.predict_proba(X_meta)
        
        # Ensure probability output matches class order
        if hasattr(self.final_estimator_, 'classes_'):
            if not np.array_equal(self.classes_, self.final_estimator_.classes_):
                # Reorder probabilities to match self.classes_
                order = np.argsort(self.final_estimator_.classes_)
                reorder = np.argsort(order)
                proba = proba[:, reorder]
                
        return proba
    
    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Compute decision function for samples in X."""
        check_is_fitted(self)
        X_meta = self.transform(X)
        return self.final_estimator_.decision_function(X_meta)
    
    def transform(self, X):
        """Transform X into meta-features."""
        check_is_fitted(self)
        return super()._transform(X)
    
    @property
    def n_features_in_(self):
        """Number of features seen during fit."""
        if hasattr(self, 'estimators_') and self.estimators_:
            return self.estimators_[0].n_features_in_
        raise AttributeError("Model not fitted yet")
    
    def get_domain_importances(self):
        """Get domain importance weights from final estimator."""
        if hasattr(self.final_estimator_, 'coef_'):
            return self.final_estimator_.coef_
        if hasattr(self.final_estimator_, 'feature_importances_'):
            return self.final_estimator_.feature_importances_
        return None
    
    def score(self, X, y_enc):
        """Return the mean accuracy on the given test data."""
        y_pred = self.predict(X)
        _, y_dec, _ = decode_domains(X, y_enc)
        return np.mean(y_pred == y_dec)