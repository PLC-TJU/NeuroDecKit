import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.base import ClassifierMixin
from sklearn.pipeline import make_pipeline
from pyriemann.classification import MDM, FgMDM, TSclassifier
from pyriemann.tangentspace import FGDA, TangentSpace



# 继承自pyriemann的TSclassifier类, 修改了fit函数中'__sample_weight'的传递条件
class TS_classifier(TSclassifier):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def fit(self, X, y, sample_weight=None):
        """Fit TSclassifier.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : TSclassifier instance
            The TSclassifier instance.
        """
        if not isinstance(self.clf, ClassifierMixin):
            raise TypeError('clf must be a ClassifierMixin')
        self.classes_ = np.unique(y)

        ts = TangentSpace(metric=self.metric, tsupdate=self.tsupdate)
        self._pipe = make_pipeline(ts, self.clf)
        sample_weight_dict = {}
        for step in self._pipe.steps:
            step_name = step[0]
            if step_name not in ['lineardiscriminantanalysis']:
                sample_weight_dict[step_name + '__sample_weight'] = sample_weight
            
        self._pipe.fit(X, y, **sample_weight_dict)
        return self
        


# 继承自pyriemann的FGDA类, 修改了_fit_lda函数中LDA的设置
# 跟MATLAB版本的FGDA类功能一致
class FGDA_model(FGDA):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    
    def _fit_lda(self, X, y, sample_weight=None):
        """Helper to fit LDA."""
        self.classes_ = np.unique(y)
        self._lda = LDA(n_components=len(self.classes_) - 1,
                        solver='eigen', # 原来是lsqr, 改成eigen
                        shrinkage='auto')

        ts = self._ts.fit_transform(X, sample_weight=sample_weight)
        self._lda.fit(ts, y)

        W = self._lda.coef_.copy()
        self._W = W.T @ np.linalg.pinv(W @ W.T) @ W
        return ts
    
class FgMDM_model(FgMDM):
    # def __init__(self,*args, **kwargs):
    #     super().__init__(*args, **kwargs)
    
    def fit(self, X, y, sample_weight=None):
        """Fit FgMDM.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : FgMDM instance
            The FgMDM instance.
        """
        self.classes_ = np.unique(y)

        self._mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)
        self._fgda = FGDA_model(metric=self.metric, tsupdate=self.tsupdate)
        cov = self._fgda.fit_transform(X, y, sample_weight=sample_weight)
        self._mdm.fit(cov, y, sample_weight=sample_weight)
        self.classes_ = self._mdm.classes_
        return self
    
    
  