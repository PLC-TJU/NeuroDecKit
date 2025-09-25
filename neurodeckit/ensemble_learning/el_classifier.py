## This is the RAVE algorithm implementation in Python.

# Matlab code: https://github.com/PLC-TJU/RAVE
# Note: This python implementation is not the same as the Matlab code.

# L. Pan et al., "Riemannian geometric and ensemble learning for decoding cross-session 
# motor imagery electroencephalography signals," J Neural Eng, vol. 20, no. 6, p. 066011, 
# Nov 22 2023, doi: 10.1088/1741-2552/ad0a01.

import numpy as np
import random
import itertools
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR

from ..pre_processing.preprocessing import Pre_Processing
from ..transfer_learning.tl_classifier import TL_Classifier
from ..transfer_learning import decode_domains
from . import StackingClassifier


class EL_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 fs_old=None,
                 fs_new=None,
                 timesets=[[0, 4], [0, 2], [1, 3], [2, 4], [0, 3], [1, 4]],
                 freqsets=[[8, 30], [8, 13], [13, 18], [18, 26], [23, 30]],
                 chansets=[None],
                 parasets=None,
                 meta_classifier=LDA(solver="lsqr", shrinkage="auto"),
                 target_domain=None,
                 n_jobs=1,
                 stack_method='auto',
                 cv=5,
                 memory=None,
                 **kwargs):
        
        # 参数初始化
        self.fs_old = fs_old
        self.fs_new = fs_new
        self.timesets = timesets
        self.freqsets = freqsets
        self.chansets = chansets
        self.parasets = parasets
        self.meta_classifier = meta_classifier
        self.target_domain = target_domain
        self.n_jobs = n_jobs
        self.stack_method = stack_method
        self.cv = cv
        self.memory = memory
        self.kwargs = kwargs
        
        # 生成参数组合
        if self.parasets is None:
            self.parasets = list(itertools.product(
                self.chansets, self.timesets, self.freqsets))
        
        # 创建基础模型
        self.base_models_ = self._create_base_models()
        
        # 创建元模型包装器
        self.stacking_models_ = [
            (f'model_{i}_{random.randint(1, 1e9)}', model)
            for i, model in enumerate(self.base_models_)
        ]
        
        # 创建堆叠分类器
        self.model = StackingClassifier(
            estimators=self.stacking_models_,
            final_estimator=self.meta_classifier,
            cv=self.cv,
            stack_method=self.stack_method,
            n_jobs=self.n_jobs,
            target_domain=self.target_domain,
        )
          
    def _create_base_models(self):
        """创建基础模型集合"""
        models = []
        for channels, time_window, freq_window in self.parasets:
            pre_processor = Pre_Processing(
                fs_old=self.fs_old,
                fs_new=self.fs_new,
                channels=channels,
                start_time=time_window[0],
                end_time=time_window[1],
                lowcut=freq_window[0],
                highcut=freq_window[1],
                memory=self.memory,
                **self.kwargs
            )
            
            model = TL_Classifier(
                pre_est=pre_processor.process,
                target_domain=self.target_domain,
                tl_mode='TL',
                memory=self.memory,
                **self.kwargs
            )
            models.append(model)
        return models
    
    def fit(self, X, y_enc):
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Training data.
        y : ndarray, shape (n_trials,)
            Extended labels for each trial.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # 解码域信息
        _, y_dec, _ = decode_domains(X, y_enc)
        self.classes_ = np.unique(y_dec)
        
        # 训练模型
        self.model.fit(X, y_enc)
        
        return self
    
    def predict(self, X):
        """Predict the labels for the test data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Test data.

        Returns
        -------
        y_pred : ndarray, shape (n_trials,)
            Predicted labels for each trial.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict the probabilities for the test data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Test data.

        Returns
        -------
        y_pred : ndarray, shape (n_trials, n_classes)
            Predicted probabilities for each trial and each class.
        """
        return self.model.predict_proba(X)
    
    def score(self, X, y_enc):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)           
            Test data.
        y : ndarray, shape (n_trials,)
            Extended labels for each trial.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """ 
        _, y_dec, _ = decode_domains(X, y_enc)
        return self.model.score(X, y_dec)
    
    def get_feature_importances(self):
        """获取元分类器的特征重要性（如果可用）"""
        if hasattr(self.meta_classifier, 'coef_'):
            return self.meta_classifier.coef_
        elif hasattr(self.meta_classifier, 'feature_importances_'):
            return self.meta_classifier.feature_importances_
        return None
    
    def get_model_configurations(self):
        """获取所有基础模型的配置信息"""
        configs = []
        for (channels, time_window, freq_window) in self.parasets:
            configs.append({
                'channels': channels,
                'time_window': time_window,
                'freq_window': freq_window
            })
        return configs
    
        
if __name__ == '__main__':
    from sklearn.model_selection import StratifiedShuffleSplit
    from ..loaddata import Dataset_Left_Right_MI
    from ..transfer_learning import TLSplitter, encode_datasets
    
    dataset_name = 'Pan2025'
    fs = 250
    dataset = Dataset_Left_Right_MI(dataset_name,fs=fs,fmin=1,fmax=40,tmin=0,tmax=4)
    subjects = dataset.subject_list[:3]

    datas, labels = [], []
    for sub in subjects:
        data, label, _ = dataset.get_data([sub])
        datas.append(data)
        labels.append(label)

    # 设置交叉验证
    n_splits=5
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=2024) 

    for sub in subjects:

            print(f"Subject {sub}...")
            target_domain = f'S{sub}'
            
            X, y_enc, domain = encode_datasets(datas, labels)
            print(f"data shape: {X.shape}, label shape: {y_enc.shape}")
            print(f"All Domain: {domain}")
            print(f'target_domain: {target_domain}')
            
            Model = EL_Classifier(target_domain=target_domain, n_jobs=-1)
            
            tl_cv = TLSplitter(target_domain=target_domain, cv=cv, no_calibration=False)
            
            acc = []
            for train, test in tl_cv.split(X, y_enc):
                X_train, y_train = X[train], y_enc[train]
                X_test, y_test = X[test], y_enc[test]
                Model.fit(X_train, y_train)
                score = Model.score(X_test, y_test)
                acc.append(score)

            print(f"Subject {sub} score: {score.mean()} +/- {score.std()}")