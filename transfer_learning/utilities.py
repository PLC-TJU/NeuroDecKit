# This file contains the implementation of various transfer learning methods.
# Author: CoreyLin
# Date: 2024.6.21

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso # Lasso 回归, L1 正则化
from sklearn.feature_selection import RFE # 递归特征消除
from sklearn.feature_selection import RFECV # 递归特征消除(结合交叉验证自动选择最佳特征数量)

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR # 逻辑回归
from sklearn.neighbors import KNeighborsClassifier as KNN # K 近邻
from sklearn.tree import DecisionTreeClassifier as DTC # 决策树
from sklearn.ensemble import RandomForestClassifier as RFC # 随机森林
from sklearn.ensemble import ExtraTreesClassifier as ETC # 极端随机森林
from sklearn.ensemble import AdaBoostClassifier as ABC # AdaBoost
from sklearn.ensemble import GradientBoostingClassifier as GBC # GradientBoosting
from sklearn.naive_bayes import GaussianNB as GNB # 高斯朴素贝叶斯
from sklearn.neural_network import MLPClassifier as MLP # 多层感知机

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace as TS
from pyriemann.spatialfilters import CSP
from pyriemann.classification import MDM, FgMDM
from pyriemann.transfer import TLDummy, TLCenter, TLStretch, TLRotate, TLClassifier

from machine_learning import TRCSP, DCPM, TRCA, SBLEST, TSclassifier
from .base import encode_datasets

__all__ = [
    'BaseEstimator',
    'ClassifierMixin',
    'clone',
    'make_pipeline',
    'Pipeline',
    'OneVsRestClassifier',
    'encode_datasets',
    'SelectKBest',
    'SelectPercentile',
    'f_classif',
    'mutual_info_classif',
    'PCA',
    'Lasso',
    'RFE',
    'RFECV',
    'SVC',
    'LDA',
    'LR',
    'KNN',
    'DTC',
    'RFC',
    'ETC',
    'ABC',
    'GBC',
    'GNB',
    'MLP',
    'Covariances',
    'TS',
    'CSP',
    'MDM',
    'FgMDM',
    'TLDummy',
    'TLCenter',
    'TLStretch',
    'TLRotate',
    'TLClassifier',
    'TRCSP',
    'DCPM',
    'TRCA',
    'SBLEST',
    'TSclassifier'
]