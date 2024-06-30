"""
TL_Classifier: Transfer Learning Classifier
Author: Pan.LC <coreylin2023@outlook.com>
Date: 2024/6/21
License: All rights reserved
"""

from .utilities import *
from utils import ensure_pipeline, combine_processes
from utils import check_pipeline_compatibility as check_compatible

class TL_Classifier(BaseEstimator, ClassifierMixin): 
    def __init__(self, dpa_method='TLDUMMY', fee_method='CSP', fes_method='MIC-K', clf_method='SVM', 
                 end_method=None, ete_method=None, pre_est=None, target_domain=None, tl_mode='TL', **kwargs):
        self.model = None
        self._tl_flag = True
        self.memory = kwargs.get('memory', None) # 缓存地址 Memory(location=None, verbose=0)
        
        self.dpa_method = dpa_method  # 域适应方法
        self.fee_method = fee_method  # 特征提取方法
        self.fes_method = fes_method  # 特征选择方法
        self.clf_method = clf_method  # 分类器方法
        self.end_method = end_method  # 末端分类器（包括特征提取、特征选择、分类器），使用样本协方差矩阵输入
        self.ete_method = ete_method  # end-to-end 分类器,使用RAW-EEG样本输入,而不是样本协方差矩阵
        self.pre_est = pre_est        # 预处理器
        
        self.target_domain = target_domain # 目标域标签 ['domain1']
        self.tl_mode = tl_mode # 迁移学习模式 ['TL', 'NOTL', 'Calibration-Free']
        # note: 
        # 'TL' means transfer learning, 
        # 'NOTL' means no transfer learning, 
        # 'Calibration-Free' means no calibration
        
        self.domain_tags = kwargs.get('domain_tags', None) # 域标签 ['domain1', 'domain2', 'domain3']
        self.domain_weight = kwargs.get('domain_weight', None) # 各个域权重 {'domain1': 0.5, 'domain2': 0.3, 'domain3': 0.2}
        
        self.csp_nfilter = kwargs.get('csp_nfilter', 8) # CSP滤波器数量
        self.fea_num = kwargs.get('fea_num', 12) # 特征数量
        self.fea_percent = kwargs.get('fea_percent', 30) # 特征数量百分比
        self.cov_estimator = kwargs.get('cov_estimator', 'lwf') # 样本协方差矩阵估计器
        
        # 预处理器
        pre_est = self.check_preest(self.pre_est)
        
        # 域适应器
        if self.ete_method is not None:
            dpa = self.check_raw_dpa(self.dpa_method)
        else:
            dpa = self.check_dpa(self.dpa_method)
        
        # 末端分类器（包括特征提取、特征选择、分类器）
        if self.ete_method is not None:
            endtoend = self.check_endtoend(self.ete_method)
            end_est = combine_processes(endtoend, memory=self.memory)
        else:
            if self.end_method is not None:
                endest = self.check_endest(self.end_method)
                end_est = combine_processes(endest, memory=self.memory)
            else:
                fee = self.check_fee(self.fee_method)
                fes = self.check_fes(self.fes_method)
                clf = self.check_clf(self.clf_method)
                end_est = combine_processes(fee, fes, clf, memory=self.memory)
        
        # 添加迁移学习框架
        if self._tl_flag:
            end_est = TLClassifier(
                target_domain=self.target_domain, 
                estimator=end_est,
                tl_mode=self.tl_mode,
                domain_weight=self.domain_weight
                )
        
        # 添加域适应器
        self.model = combine_processes(pre_est, dpa, end_est, memory=self.memory)

    def check_dpa(self, dpa):
        dpa = 'None' if dpa is None else dpa
        prealignments = {
            'TLDUMMY': make_pipeline(
                Covariances(estimator=self.cov_estimator),
                TLDummy(),
            ),
            'EA': make_pipeline(
                Covariances(estimator=self.cov_estimator),
                TLCenter(target_domain=self.target_domain, metric='euclid'),
            ),
            'RA': make_pipeline(
                Covariances(estimator=self.cov_estimator),
                TLCenter(target_domain=self.target_domain, metric='riemann'),
            ),
            'RPA': make_pipeline(
                Covariances(estimator=self.cov_estimator),
                TLCenter(target_domain=self.target_domain),
                TLStretch(
                    target_domain=self.target_domain,
                    final_dispersion=1,
                    centered_data=True,
                ),
                TLRotate(target_domain=self.target_domain, metric='riemann'),
            ),
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
    
    def check_raw_dpa(self, dpa):
        dpa = 'None' if dpa is None else dpa
        prealignments = {
            'TLDUMMY': make_pipeline(
                TLDummy(),
            ),
            'EA': make_pipeline(
                RCT(target_domain=self.target_domain, metric='euclid'),
            ),
            'RA': make_pipeline(
                RCT(target_domain=self.target_domain, metric='riemann'),
            ),
            'RPA': make_pipeline(
                RCT(target_domain=self.target_domain, metric='riemann'),
                STR(target_domain=self.target_domain, metric='riemann'),
                ROT(target_domain=self.target_domain, metric='riemann'),
            ),
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
        fee = 'None' if fee is None else fee
        feature_extractions = {
            'NONE':   [],
            'CSP':    ('csp',   CSP(nfilter=self.csp_nfilter, metric='euclid')),
            'TRCSP':  ('trcsp', TRCSP(nfilter=self.csp_nfilter, metric='euclid')), # only for 2 classes
            'MDM':    ('mdm',   MDM()),
            'FGMDM':  ('fgmdm', FgMDM()),
            'TS':     ('ts',    TS()),
        }
        if callable(fee):
            pass
        elif fee.upper() in feature_extractions.keys():
            if 'MDM' in fee.upper() or 'MDRM' in fee.upper():
                self.fea_percent = 100 # 特征数量百分比
                self.fea_num = 100 # 特征最大数量
            if 'MEKT' in fee.upper() or 'MDWM' in fee.upper():
                self._tl_flag = False # 关闭迁移学习框架
            if ('MEKT' in fee.upper() or 'MDWM' in fee.upper()) and self.tl_mode != 'TL':
                raise ValueError('%s is only available in transfer learning (tl_mode = TL) mode !' % fee.upper())
            # Map the corresponding estimator
            fee = feature_extractions[fee.upper()]
        elif check_compatible(fee):
            pass
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (fee, (' , ').join(feature_extractions.keys())))
        return fee
    
    def check_fes(self, fes):
        fes = 'None' if fes is None else fes
        feature_selections = {
            'NONE':      Pipeline(steps=[]),
            'ANOVA-K':   SelectKBest(f_classif, k=self.fea_num), # 基于F值的特征选择
            'ANOVA-P':   SelectPercentile(f_classif, percentile=self.fea_percent),
            'MIC-K':     SelectKBest(mutual_info_classif, k=self.fea_num), # 基于互信息的特征选择
            'MIC-P':     SelectPercentile(mutual_info_classif, percentile=self.fea_percent),
            'PCA':       PCA(n_components=0.9), # 基于PCA的特征降维
            'LASSO':     Lasso(alpha=0.01), # 基于Lasso回归的特征选择 (L1正则化)
            'RFE':       RFE(estimator=LR(), n_features_to_select=self.fea_num), # 基于递归特征消除的特征选择
            'RFECV':     RFECV(estimator=LR(), step=1, cv=5, scoring='accuracy'), # 基于递归特征消除的交叉验证特征选择
        }
        if callable(fes):
            pass
        elif fes.upper() in feature_selections.keys():
            # Map the corresponding estimator
            fes = feature_selections[fes.upper()]
        elif check_compatible(fes):
            pass
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (fes, (' , ').join(feature_selections.keys())))
        return fes
    
    def check_clf(self, clf):
        clf = 'None' if clf is None else clf
        classifiers = {
            'NONE': Pipeline(steps=[]), # 空分类器
            'SVM':  ('svm', SVC(C=1, kernel='linear')),#支持向量机
            'LDA':  ('lda', LDA(solver='eigen', shrinkage='auto')), # 线性判别分析, **注意：LDA没有sample_weight参数
            'LR':   ('lr',  LR()), # 逻辑回归
            'KNN':  ('knn', KNN(n_neighbors=5)), # K近邻, **注意：KNN没有sample_weight参数
            'DTC':  ('dtc', DTC(min_samples_split=2)), # 决策树分类器
            'RFC':  ('rfc', RFC(n_estimators=100)), # 随机森林分类器
            'ETC':  ('etc', ETC(n_estimators=100)), # 极限随机树分类器
            'ABC':  ('abc', ABC(estimator=None, n_estimators=50, algorithm='SAMME')), # AdaBoost分类器
            'GBC':  ('gbc', GBC(n_estimators=100)), # GradientBoosting分类器
            'GNB':  ('gnb', GNB()), # 高斯朴素贝叶斯分类器
            'MLP':  ('mlp', MLP(hidden_layer_sizes=(100,), max_iter=1000, alpha=0.0001, solver='adam')), # 多层感知机, **注意：MLP没有sample_weight参数
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
    
    def check_endest(self, est, n_estimators=50, algorithm='SAMME'):
        est = 'None' if est is None else est
        estimators = {
            'NONE':        [],
            'RKNN':        ('rknn',      RKNN(n_neighbors=5, metric='riemann')), 
            'RKSVM':       ('rksvm',     RKSVM(C=1, metric='riemann')),
            'ABC-MDM':     ('abc-mdm',   ABC(estimator=MDM(), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-FGMDM':   ('abc-fgmdm', ABC(estimator=FgMDM(), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-LDA':  ('abc-tslda', ABC(TSclassifier(clf=LDA()), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-LR':   ('abc-tsglm', ABC(TSclassifier(clf=LR()), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-GLM':  ('abc-tsglm', ABC(TSclassifier(clf=LR()), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-SVM':  ('abc-tssvm', ABC(TSclassifier(clf=SVC()), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-MIC-K-LDA': ('abc-ts-mic-k-lda', ABC(TSclassifier(clf=make_pipeline(
                SelectKBest(mutual_info_classif, k=self.fea_num), LDA()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-MIC-K-LR':  ('abc-ts-mic-k-lr', ABC(TSclassifier(clf=make_pipeline(
                SelectKBest(mutual_info_classif, k=self.fea_num), LR()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-MIC-K-SVM': ('abc-ts-mic-k-svm', ABC(TSclassifier(clf=make_pipeline(
                SelectKBest(mutual_info_classif, k=self.fea_num), SVC()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-MIC-P-LDA': ('abc-ts-mic-p-lda', ABC(TSclassifier(clf=make_pipeline(
                SelectPercentile(mutual_info_classif, percentile=self.fea_percent), LDA()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-MIC-P-LR':  ('abc-ts-mic-p-lr', ABC(TSclassifier(clf=make_pipeline(
                SelectPercentile(mutual_info_classif, percentile=self.fea_percent), LR()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-MIC-P-SVM': ('abc-ts-mic-p-svm', ABC(TSclassifier(clf=make_pipeline(
                SelectPercentile(mutual_info_classif, percentile=self.fea_percent), SVC()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-PCA-LDA':   ('abc-ts-pca-lda', ABC(TSclassifier(clf=make_pipeline(
                PCA(n_components=0.9), LDA()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-PCA-LR':    ('abc-ts-pca-lr', ABC(TSclassifier(clf=make_pipeline(
                PCA(n_components=0.9), LR()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-PCA-SVM':   ('abc-ts-pca-svm', ABC(TSclassifier(clf=make_pipeline(
                PCA(n_components=0.9), SVC()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-LASSO-LDA': ('abc-ts-lasso-lda', ABC(TSclassifier(clf=make_pipeline(
                Lasso(alpha=0.01), LDA()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-LASSO-LR':  ('abc-ts-lasso-lr', ABC(TSclassifier(clf=make_pipeline(
                Lasso(alpha=0.01), LR()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-LASSO-SVM': ('abc-ts-lasso-svm', ABC(TSclassifier(clf=make_pipeline(
                Lasso(alpha=0.01), SVC()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-RFE-LDA':   ('abc-ts-rfe-lda', ABC(TSclassifier(clf=make_pipeline(
                RFE(estimator=LDA(), n_features_to_select=self.fea_num), LDA()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-RFE-LR':    ('abc-ts-rfe-lr', ABC(TSclassifier(clf=make_pipeline(
                RFE(estimator=LR(), n_features_to_select=self.fea_num), LR()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-RFE-SVM':   ('abc-ts-rfe-svm', ABC(TSclassifier(clf=make_pipeline(
                RFE(estimator=SVC(), n_features_to_select=self.fea_num), SVC()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-RFECV-LDA': ('abc-ts-rfecv-lda', ABC(TSclassifier(clf=make_pipeline(
                RFECV(estimator=LDA(), step=1, cv=5, scoring='accuracy'), LDA()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-RFECV-LR':  ('abc-ts-rfecv-lr', ABC(TSclassifier(clf=make_pipeline(
                RFECV(estimator=LR(), step=1, cv=5, scoring='accuracy'), LR()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-RFECV-SVM': ('abc-ts-rfecv-svm', ABC(TSclassifier(clf=make_pipeline(
                RFECV(estimator=SVC(), step=1, cv=5, scoring='accuracy'), SVC()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-ANOVA-K-LDA': ('abc-ts-anova-k-lda', ABC(TSclassifier(clf=make_pipeline(
                SelectKBest(f_classif, k=self.fea_num), LDA()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-ANOVA-K-LR':  ('abc-ts-anova-k-lr', ABC(TSclassifier(clf=make_pipeline(
                SelectKBest(f_classif, k=self.fea_num), LR()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-ANOVA-K-SVM': ('abc-ts-anova-k-svm', ABC(TSclassifier(clf=make_pipeline(
                SelectKBest(f_classif, k=self.fea_num), SVC()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-ANOVA-P-LDA': ('abc-ts-anova-p-lda', ABC(TSclassifier(clf=make_pipeline(
                SelectPercentile(f_classif, percentile=self.fea_percent), LDA()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-ANOVA-P-LR':  ('abc-ts-anova-p-lr', ABC(TSclassifier(clf=make_pipeline(
                SelectPercentile(f_classif, percentile=self.fea_percent), LR()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'ABC-TS-ANOVA-P-SVM': ('abc-ts-anova-p-svm', ABC(TSclassifier(clf=make_pipeline(
                SelectPercentile(f_classif, percentile=self.fea_percent), SVC()), memory=self.memory), n_estimators=n_estimators, algorithm=algorithm)),
            'MDWM':       ('mdwm',      MDWM(domain_tradeoff=0.5, target_domain=self.target_domain)),# 本身包括迁移学习框架，仅适用于迁移学习
            'MEKT':       ('mekt-lda',  MEKT(target_domain=self.target_domain)), # 本身包括迁移学习框架，仅适用于迁移学习
            'MEKT-LDA':   ('mekt-lda',  MEKT(target_domain=self.target_domain, estimator=LDA())), 
            'MEKT-LR':    ('mekt-lr',   MEKT(target_domain=self.target_domain, estimator=LR())),
            'MEKT-SVM':   ('mekt-svm',  MEKT(target_domain=self.target_domain, estimator=SVC())),
            'MEKT-MLP':   ('mekt-mlp',  MEKT(target_domain=self.target_domain, estimator=MLP())),
            'MEKT-MIC-K-LDA':   ('mekt-mic-k-lda',  MEKT(target_domain=self.target_domain, estimator=LDA(), selector=SelectKBest(mutual_info_classif, k=self.fea_num))),
            'MEKT-MIC-K-LR':    ('mekt-mic-k-lr',   MEKT(target_domain=self.target_domain, estimator=LR(), selector=SelectKBest(mutual_info_classif, k=self.fea_num))),
            'MEKT-MIC-K-SVM':   ('mekt-mic-k-svm',  MEKT(target_domain=self.target_domain, estimator=SVC(), selector=SelectKBest(mutual_info_classif, k=self.fea_num))),
            'MEKT-MIC-K-MLP':   ('mekt-mic-k-mlp',  MEKT(target_domain=self.target_domain, estimator=MLP(), selector=SelectKBest(mutual_info_classif, k=self.fea_num))),
            'MEKT-MIC-P-LDA':   ('mekt-mic-p-lda',  MEKT(target_domain=self.target_domain, estimator=LDA(), selector=SelectPercentile(mutual_info_classif, percentile=self.fea_percent))),
            'MEKT-MIC-P-LR':    ('mekt-mic-p-lr',   MEKT(target_domain=self.target_domain, estimator=LR(), selector=SelectPercentile(mutual_info_classif, percentile=self.fea_percent))),
            'MEKT-MIC-P-SVM':   ('mekt-mic-p-svm',  MEKT(target_domain=self.target_domain, estimator=SVC(), selector=SelectPercentile(mutual_info_classif, percentile=self.fea_percent))),
            'MEKT-MIC-P-MLP':   ('mekt-mic-p-mlp',  MEKT(target_domain=self.target_domain, estimator=MLP(), selector=SelectPercentile(mutual_info_classif, percentile=self.fea_percent))),
            'MEKT-ANOVA-K-LDA':   ('mekt-anova-k-lda',  MEKT(target_domain=self.target_domain, estimator=LDA(), selector=SelectKBest(f_classif, k=self.fea_num))),
            'MEKT-ANOVA-K-LR':    ('mekt-anova-k-lr',   MEKT(target_domain=self.target_domain, estimator=LR(), selector=SelectKBest(f_classif, k=self.fea_num))),
            'MEKT-ANOVA-K-SVM':   ('mekt-anova-k-svm',  MEKT(target_domain=self.target_domain, estimator=SVC(), selector=SelectKBest(f_classif, k=self.fea_num))),
            'MEKT-ANOVA-K-MLP':   ('mekt-anova-k-mlp',  MEKT(target_domain=self.target_domain, estimator=MLP(), selector=SelectKBest(f_classif, k=self.fea_num))),
            'MEKT-ANOVA-P-LDA':   ('mekt-anova-p-lda',  MEKT(target_domain=self.target_domain, estimator=LDA(), selector=SelectPercentile(f_classif, percentile=self.fea_percent))),
            'MEKT-ANOVA-P-LR':    ('mekt-anova-p-lr',   MEKT(target_domain=self.target_domain, estimator=LR(), selector=SelectPercentile(f_classif, percentile=self.fea_percent))),
            'MEKT-ANOVA-P-SVM':   ('mekt-anova-p-svm',  MEKT(target_domain=self.target_domain, estimator=SVC(), selector=SelectPercentile(f_classif, percentile=self.fea_percent))),
            'MEKT-ANOVA-P-MLP':   ('mekt-anova-p-mlp',  MEKT(target_domain=self.target_domain, estimator=MLP(), selector=SelectPercentile(f_classif, percentile=self.fea_percent))),
            'MEKT-PCA-LDA':   ('mekt-pca-lda',  MEKT(target_domain=self.target_domain, estimator=LDA(), selector=PCA(n_components=0.9))),
            'MEKT-PCA-LR':    ('mekt-pca-lr',   MEKT(target_domain=self.target_domain, estimator=LR(), selector=PCA(n_components=0.9))),
            'MEKT-PCA-SVM':   ('mekt-pca-svm',  MEKT(target_domain=self.target_domain, estimator=SVC(), selector=PCA(n_components=0.9))),
            'MEKT-PCA-MLP':   ('mekt-pca-mlp',  MEKT(target_domain=self.target_domain, estimator=MLP(), selector=PCA(n_components=0.9))),
            'MEKT-LASSO-LDA':   ('mekt-lasso-lda',  MEKT(target_domain=self.target_domain, estimator=LDA(), selector=Lasso(alpha=0.01))),
            'MEKT-LASSO-LR':    ('mekt-lasso-lr',   MEKT(target_domain=self.target_domain, estimator=LR(), selector=Lasso(alpha=0.01))),
            'MEKT-LASSO-SVM':   ('mekt-lasso-svm',  MEKT(target_domain=self.target_domain, estimator=SVC(), selector=Lasso(alpha=0.01))),
            'MEKT-LASSO-MLP':   ('mekt-lasso-mlp',  MEKT(target_domain=self.target_domain, estimator=MLP(), selector=Lasso(alpha=0.01))),
            'MEKT-RFE-LDA':   ('mekt-rfe-lda',  MEKT(target_domain=self.target_domain, estimator=LDA(), selector=RFE(estimator=LR(), n_features_to_select=self.fea_num))),
            'MEKT-RFE-LR':    ('mekt-rfe-lr',   MEKT(target_domain=self.target_domain, estimator=LR(), selector=RFE(estimator=LR(), n_features_to_select=self.fea_num))),
            'MEKT-RFE-SVM':   ('mekt-rfe-svm',  MEKT(target_domain=self.target_domain, estimator=SVC(), selector=RFE(estimator=LR(), n_features_to_select=self.fea_num))),
            'MEKT-RFE-MLP':   ('mekt-rfe-mlp',  MEKT(target_domain=self.target_domain, estimator=MLP(), selector=RFE(estimator=LR(), n_features_to_select=self.fea_num))),
            'MEKT-RFECV-LDA':   ('mekt-rfecv-lda',  MEKT(target_domain=self.target_domain, estimator=LDA(), selector=RFECV(estimator=LR(), step=1, cv=5))),
            'MEKT-RFECV-LR':    ('mekt-rfecv-lr',   MEKT(target_domain=self.target_domain, estimator=LR(), selector=RFECV(estimator=LR(), step=1, cv=5))),
            'MEKT-RFECV-SVM':   ('mekt-rfecv-svm',  MEKT(target_domain=self.target_domain, estimator=SVC(), selector=RFECV(estimator=LR(), step=1, cv=5))),
            'MEKT-RFECV-MLP':   ('mekt-rfecv-mlp',  MEKT(target_domain=self.target_domain, estimator=MLP(), selector=RFECV(estimator=LR(), step=1, cv=5))),
        }
        if callable(est):
            pass
        elif est.upper() in estimators.keys():
            if 'MEKT' in est.upper() or 'MDWM' in est.upper():
                self._tl_flag = False # 关闭迁移学习框架
            if ('MEKT' in est.upper() or 'MDWM' in est.upper()) and self.tl_mode != 'TL':
                raise ValueError('%s is only available in transfer learning (tl_mode = TL) mode !' % est.upper())
            # Map the corresponding estimator
            est = estimators[est.upper()]
        elif check_compatible(est):
            pass
        else:
            # raise an error
            raise ValueError(
                """%s is not an valid estimator ! Valid estimators are : %s or a
                callable function""" % (est, (' , ').join(estimators.keys())))
        return est
    
    def check_endtoend(self, endtoend):
        endtoend = 'None' if endtoend is None else endtoend
        endtoends = {
            'NONE':   [],
            'TRCA':   ('trca',   TRCA(n_components=6)),
            'DCPM':   ('dcpm',   DCPM(n_components=6)),
            'SBLEST': ('sblest', OneVsRestClassifier(SBLEST(K=2, tau=1, Epoch=2000, epoch_print=0))),
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
    
    def check_preest(self, pre_est):
        
        return ensure_pipeline(pre_est)

    def fit(self, X, y_enc):
        
        self.model.fit(X, y_enc)
         
        return self

    def predict(self, X):
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        
        return self.model.predict_proba(X)
    
    def score(self, X, y_enc):
        
        return self.model.score(X, y_enc)
        
        