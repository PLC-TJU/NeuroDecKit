"""
TL_Classifier: Transfer Learning Classifier
Author: Pan.LC <coreylin2023@outlook.com>
Date: 2024/6/21
License: MIT License

"""

from .utilities import *

class TL_Classifier(BaseEstimator, ClassifierMixin): 
    def __init__(self, dpa_method='noTL', fee_method='CSP', fes_method='MIC-K', clf_method='SVM', 
                 endtoend=None, end_est=None, target_domain=None, **kwargs):
        self.model = None
        self.memory = kwargs.get('memory', None) # 缓存地址 Memory(location=None, verbose=0)
        
        self.dpa_method = dpa_method  # 域适应方法
        self.fee_method = fee_method  # 特征提取方法
        self.fes_method = fes_method  # 特征选择方法
        self.clf_method = clf_method  # 分类器方法
        self.endtoend = endtoend     # 端到端分类器, 不适合迁移学习
        self.end_est = end_est       # 末端分类器（包括特征提取、特征选择、分类器）
        
        self.target_domain = target_domain # 目标域标签 ['domain1']
        self.domain_tags = kwargs.get('domain_tags', None) # 域标签 ['domain1', 'domain2', 'domain3']
        self.domain_weight = kwargs.get('domain_weight', None) # 各个域权重 {'domain1': 0.5, 'domain2': 0.3, 'domain3': 0.2}
        
        self.fea_num = kwargs.get('fea_num', 12) # 特征数量 
        self.fea_percent = kwargs.get('fea_percent', 0.1) # 特征百分比 
        
        # 组装分类流程
        if self.endtoend is not None:
            endtoend = self.check_endtoend(self.endtoend)
            estimator = make_pipeline(endtoend, memory=self.memory)
        else:
            dpa = self.check_dpa(self.dpa_method)
            if self.end_est is not None:
                end_est = self.check_end_est(self.end_est)
                estimator = make_pipeline(dpa, end_est, memory=self.memory)
            else:
                fee = self.check_fee(self.fee_method)
                fes = self.check_fes(self.fes_method)
                clf = self.check_clf(self.clf_method)
                estimator = make_pipeline(dpa, fee, fes, clf, memory=self.memory)
        
        # 迁移学习分类器
        if dpa_method != 'NOTL' and endtoend is None:
            self.model = TLClassifier(target_domain=self.target_domain, estimator=estimator, domain_weight=self.domain_weight)
        else:
            self.model = estimator
        
    def check_dpa(self, dpa):
        prealignments = {
            'NOTL': Pipeline(steps=[]),
            'TLDUMMY': make_pipeline(
                Covariances(estimator='lwf'),
                TLDummy(),
            ),
            'EA': make_pipeline(
                Covariances(estimator='lwf'),
                TLCenter(target_domain=self.target_domain, metric='euclid'),
            ),
            'RA': make_pipeline(
                Covariances(estimator='lwf'),
                TLCenter(target_domain=self.target_domain, metric='riemann'),
            ),
            'RPA': make_pipeline(
                Covariances(estimator='lwf'),
                TLCenter(target_domain=self.target_domain),
                TLStretch(
                    target_domain=self.target_domain,
                    final_dispersion=1,
                    centered_data=True,
                ),
                TLRotate(target_domain=self.target_domain, metric='euclid'),
            )
        }
        if callable(dpa):
            pass
        elif dpa.upper() in prealignments.keys():
            # Map the corresponding estimator
            dpa = prealignments[dpa.upper()]
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (dpa, (' , ').join(prealignments.keys())))
        return dpa      
    
    def check_fee(self, fee):
        transformers = {
            'CSP':    CSP(nfilter=8, metric='euclid'),
            'TRCSP':  TRCSP(nfilter=8, metric='euclid'),
            'MDM':    MDM(),
            'MDRM':   MDM(metric='riemann'),
            'FGMDM':  FgMDM(),
            'FGMDRM': FgMDM(metric='riemann'),
            'TS':     TS(),
        }
        if callable(fee):
            pass
        elif fee.upper() in transformers.keys():
            # Map the corresponding estimator
            fee = transformers[fee.upper()]
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (fee, (' , ').join(transformers.keys())))
        return fee
    
    def check_fes(self, fes):
        fes = 'None' if fes is None else fes
        feature_selection = {
            'None':      Pipeline(steps=[]),
            'ANOVA-F-K': SelectKBest(f_classif, k=self.fea_num), # 基于F值的特征选择
            'ANOVA-F-P': SelectPercentile(f_classif, percentile=self.fea_percent),
            'MIC-K':     SelectKBest(mutual_info_classif, k=self.fea_num), # 基于互信息的特征选择
            'MIC-P':     SelectPercentile(mutual_info_classif, percentile=self.fea_percent),
            'PCA-K':     PCA(n_components=self.fea_num), # 基于PCA的特征降维
            'Lasso':     Lasso(alpha=0.01), # 基于Lasso回归的特征选择
            'L1':        Lasso(alpha=0.01),
            'RFE':       RFE(estimator=SVC(), n_features_to_select=self.fea_num), # 基于递归特征消除的特征选择
            'RFECV':     RFECV(estimator=SVC(), step=1, cv=5, scoring='accuracy'),     
        }
        if callable(fes):
            pass
        elif fes.upper() in feature_selection.keys():
            # Map the corresponding estimator
            fes = feature_selection[fes.upper()]
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (fes, (' , ').join(feature_selection.keys())))
        return fes
    
    def check_clf(self, clf):
        clf = 'None' if clf is None else clf
        classifiers = {
            'None': Pipeline(steps=[]),
            'SVM':  SVC(),
            'LDA':  LDA(solver='eigen', shrinkage='auto'),
            'LR':   LR(),
            'KNN':  KNN(n_neighbors=5),
            'DTC':  DTC(min_samples_split=2),
            'RFC':  RFC(n_estimators=100),
            'ETC':  ETC(n_estimators=100),
            'ABC':  ABC(estimator=None, n_estimators=50, algorithm='SAMME'),
            'GBC':  GBC(n_estimators=100),
            'GNB':  GNB(),
            'MLP':  MLP(hidden_layer_sizes=(50,), max_iter=1000, alpha=0.0001, solver='adam'),
        }
        if callable(clf):
            pass
        elif clf.upper() in classifiers.keys():
            # Map the corresponding estimator
            clf = classifiers[clf.upper()]       
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a    
                callable function""" % (clf, (' , ').join(classifiers.keys())))
        return clf
    
    def check_endtoend(self, endtoend):
        endtoends = {
            'TRCA':      TRCA(n_components=6),
            'DCPM':      DCPM(n_components=6),
            'SBLEST':    OneVsRestClassifier(SBLEST(K=2, tau=1, Epoch=5000, epoch_print=0)),
        }
        if callable(endtoend):
            pass
        elif endtoend.upper() in endtoends.keys():
            # Map the corresponding estimator
            endtoend = endtoends[endtoend.upper()]
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (endtoend, (' , ').join(endtoends.keys())))
        return endtoend
    
    def check_end_est(self, est, n_estimators=50, algorithm='SAMME'):
        estimator = {
            'ABC-MDM':   ABC(estimator=MDM(), n_estimators=n_estimators, algorithm=algorithm),
            'ABC-MDRM':  ABC(estimator=MDM(metric='riemann'), n_estimators=n_estimators, algorithm=algorithm),
            'ABC-FGMDM': ABC(estimator=FgMDM(), n_estimators=n_estimators, algorithm=algorithm),
            'ABC-FGMDRM':ABC(estimator=FgMDM(metric='riemann'), n_estimators=n_estimators, algorithm=algorithm),
            'ABC-TSLDA': ABC(TSclassifier(clf=LDA()), n_estimators=n_estimators, algorithm=algorithm),
            'ABC-TSLR':  ABC(TSclassifier(clf=LR()), n_estimators=n_estimators, algorithm=algorithm),
            'ABC-TSGLM': ABC(TSclassifier(clf=LR()), n_estimators=n_estimators, algorithm=algorithm),
            'ABC-TSSVM': ABC(TSclassifier(clf=SVC()), n_estimators=n_estimators, algorithm=algorithm),
        }
        if callable(est):
            est = ABC(estimator=est, n_estimators=n_estimators, algorithm=algorithm)
        elif est.upper() in estimator.keys():
            # Map the corresponding estimator
            est = estimator[est.upper()]
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (est, (' , ').join(estimator.keys())))
        return est

    def fit(self, X, y):
        
        # 编码标签
        X, y_enc= encode_datasets(X, y, domain_tags=self.domain_tags)
        
        self.model.fit(X, y_enc)
         
        return self

    def predict(self, X):    
        
        return self.model.predict(X)
        
        