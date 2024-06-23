"""
TL_Classifier: Transfer Learning Classifier
Author: Pan.LC <coreylin2023@outlook.com>
Date: 2024/6/21
License: MIT License

"""

from .utilities import *
from utils import ensure_pipeline, combine_processes
from utils import check_pipeline_compatibility as check_compatible

class TL_Classifier(BaseEstimator, ClassifierMixin): 
    def __init__(self, dpa_method='noTL', fee_method='CSP', fes_method='MIC-K', clf_method='SVM', 
                 endtoend=None, end_est=None, pre_est=None, target_domain=None, **kwargs):
        self.model = None
        self.memory = kwargs.get('memory', None) # 缓存地址 Memory(location=None, verbose=0)
        
        self.dpa_method = dpa_method  # 域适应方法
        self.fee_method = fee_method  # 特征提取方法
        self.fes_method = fes_method  # 特征选择方法
        self.clf_method = clf_method  # 分类器方法
        self.endtoend = endtoend     # 端到端分类器, 不适合迁移学习
        self.end_est = end_est       # 末端分类器（包括特征提取、特征选择、分类器）
        self.pre_est = pre_est       # 预处理器
        
        self.target_domain = target_domain # 目标域标签 ['domain1']
        self.domain_tags = kwargs.get('domain_tags', None) # 域标签 ['domain1', 'domain2', 'domain3']
        self.domain_weight = kwargs.get('domain_weight', None) # 各个域权重 {'domain1': 0.5, 'domain2': 0.3, 'domain3': 0.2}
        
        self.csp_nfilter = kwargs.get('csp_nfilter', 8) # CSP滤波器数量 
        self.fea_num = kwargs.get('fea_num', 12) # 特征数量 
        self.fea_percent = kwargs.get('fea_percent', 0.1) # 特征百分比 
        
        # 预处理器
        pre_est = self.check_pre_est(self.pre_est)
        
        # 域适应器
        dpa = self.check_dpa(self.dpa_method)
        
        # 末端分类器
        if self.endtoend is not None:
            endtoend = self.check_endtoend(self.endtoend)
            self.model = combine_processes(pre_est, endtoend, memory=self.memory)
        else:
            if self.end_est is not None:
                end_est = self.check_end_est(self.end_est)
            else:
                fee = self.check_fee(self.fee_method)
                fes = self.check_fes(self.fes_method)
                clf = self.check_clf(self.clf_method)
                end_est = combine_processes(fee, fes, clf, memory=self.memory)
        
            # 添加迁移学习框架
            if dpa_method.upper() != 'NOTL':
                end_est = TLClassifier(
                    target_domain=self.target_domain, 
                    estimator=end_est, 
                    domain_weight=self.domain_weight
                    )
            
            # 添加域适应器
            self.model = combine_processes(pre_est, dpa, end_est, memory=self.memory)

    def check_dpa(self, dpa):
        prealignments = {
            'NOTL': make_pipeline(
                Covariances(estimator='lwf')
            ),
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
        elif check_compatible(dpa):
            pass
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (dpa, (' , ').join(prealignments.keys())))
        return dpa      
    
    def check_fee(self, fee):
        transformers = {
            'CSP':    ('csp', CSP(nfilter=self.csp_nfilter, metric='euclid')),
            'TRCSP':  ('trcsp', TRCSP(nfilter=self.csp_nfilter, metric='euclid')), # only for 2 classes
            'MDM':    ('mdm', MDM()),
            'MDRM':   ('mdm', MDM(metric='riemann')), # 与MDM相同
            'MDWM':   ('mdwm', MDWM(domain_tradeoff=0.5, target_domain=self.target_domain)),# 仅适用于迁移学习
            'MEKT':   ('mekt', MEKT(target_domain=self.target_domain)), # 仅适用于迁移学习
            'FGMDM':  ('fgmdm', FgMDM()),
            'FGMDRM': ('fgmdm', FgMDM(metric='riemann')), # 与FGMDM相同
            'TS':     ('ts', TS()),
            'FGDA':   ('fgda', FGDA()), 
        }
        if callable(fee):
            pass
        elif fee.upper() in transformers.keys():
            if fee.upper() in ['MDWM', 'MEKT'] and self.dpa_method.upper() == 'NOTL':
                raise ValueError(
                    """%s is not an valid estimator for NOTL !""" % (fee.upper()))   
            # Map the corresponding estimator
            fee = transformers[fee.upper()]
        elif check_compatible(fee):
            pass
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
            'L1':        Lasso(alpha=0.01), # 与Lasso相同
            'RFE':       RFE(estimator=SVC(), n_features_to_select=self.fea_num), # 基于递归特征消除的特征选择
            'RFECV':     RFECV(estimator=SVC(), step=1, cv=5, scoring='accuracy'), # 基于递归特征消除的交叉验证特征选择
        }
        if callable(fes):
            pass
        elif fes.upper() in feature_selection.keys():
            # Map the corresponding estimator
            fes = feature_selection[fes.upper()]
        elif check_compatible(fes):
            pass
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (fes, (' , ').join(feature_selection.keys())))
        return fes
    
    def check_clf(self, clf):
        clf = 'None' if clf is None else clf
        classifiers = {
            'None': Pipeline(steps=[]), # 空分类器
            'SVM':  ('svm', SVC(C=1, kernel='linear')),#支持向量机
            'LDA':  ('lda', LDA(solver='eigen', shrinkage='auto')), # 线性判别分析
            'LR':   ('lr', LR()), # 逻辑回归
            'KNN':  ('knn', KNN(n_neighbors=5)), # K近邻
            'DTC':  ('dtc', DTC(min_samples_split=2)), # 决策树分类器
            'RFC':  ('rfc', RFC(n_estimators=100)), # 随机森林分类器
            'ETC':  ('etc', ETC(n_estimators=100)), # 极限随机树分类器
            'ABC':  ('abc', ABC(estimator=None, n_estimators=50, algorithm='SAMME')), # AdaBoost分类器
            'GBC':  ('gbc', GBC(n_estimators=100)), # GradientBoosting分类器
            'GNB':  ('gnb', GNB()), # 高斯朴素贝叶斯分类器
            'MLP':  ('mlp', MLP(hidden_layer_sizes=(50,), max_iter=1000, alpha=0.0001, solver='adam')), # 多层感知机
        }
        if callable(clf):
            pass
        elif clf.upper() in classifiers.keys():
            # Map the corresponding estimator
            clf = classifiers[clf.upper()]   
        elif check_compatible(clf):
            pass
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a    
                callable function""" % (clf, (' , ').join(classifiers.keys())))
        return clf
    
    def check_endtoend(self, endtoend):
        endtoends = {
            'TRCA':   ('trca', TRCA(n_components=6)),
            'DCPM':   ('dcpm', DCPM(n_components=6)),
            'SBLEST': ('sblest', OneVsRestClassifier(SBLEST(K=2, tau=1, Epoch=5000, epoch_print=0))),
        }
        if callable(endtoend):
            pass
        elif endtoend.upper() in endtoends.keys():
            # Map the corresponding estimator
            endtoend = endtoends[endtoend.upper()]
        elif check_compatible(endtoend):
            pass
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (endtoend, (' , ').join(endtoends.keys())))
        return endtoend
    
    def check_end_est(self, est, n_estimators=50, algorithm='SAMME'):
        estimator = {
            'RKNN':       ('rknn', RKNN(n_neighbors=5, metric='riemann')), 
            'RKSVM':      ('rksvm', RKSVM(C=1, metric='riemann')),
            'ABC-MDM':    ('abc-mdm', ABC(estimator=MDM(), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-MDRM':   ('abc-mdm', ABC(estimator=MDM(metric='riemann'), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-FGMDM':  ('abc-fgmdm', ABC(estimator=FgMDM(), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-FGMDRM': ('abc-fgmdm', ABC(estimator=FgMDM(metric='riemann'), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TSLDA':  ('abc-tslda', ABC(TSclassifier(clf=LDA()), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TSLR':   ('abc-tsglm', ABC(TSclassifier(clf=LR()), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TSGLM':  ('abc-tsglm', ABC(TSclassifier(clf=LR()), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TSSVM':  ('abc-tssvm', ABC(TSclassifier(clf=SVC()), n_estimators=n_estimators, algorithm=algorithm)),
        }
        if callable(est):
            pass
        elif est.upper() in estimator.keys():
            # Map the corresponding estimator
            est = estimator[est.upper()]
        elif check_compatible(est):
            pass
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (est, (' , ').join(estimator.keys())))
        return est
    
    def check_pre_est(self, pre_est):
       
        return ensure_pipeline(pre_est)

    def fit(self, X, y_enc):
        
        self.model.fit(X, y_enc)
         
        return self

    def predict(self, X):    
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        
        return self.model.score(X, y)
        
        