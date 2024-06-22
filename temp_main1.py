import numpy as np
from pyriemann.estimation import Covariances
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from joblib import Memory

from loaddata import Dataset_Left_Right_MI
from machine_learning.classifier import svc,lda,lr,knn,dtc,rfc,etc,abc,gbc,gnb,mlp
from machine_learning import RiemannCSP as CSP
from machine_learning import TS

# 加载数据
dataset_name = 'Pan2023'
fs = 250
freqband = [8,30]
datapath = r'E:\工作进展\小论文2023会议\数据处理python\datasets'
dataset = Dataset_Left_Right_MI(dataset_name,fs,fmin=freqband[0],fmax=freqband[1],tmin=0,tmax=4,path=datapath)
data, label = dataset.get_data([1])

# 设置交叉验证
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# 设置缓存目录
cachedir = 'my_cache_directory'
memory = Memory(cachedir, verbose=0)

data = Covariances(estimator='lwf').transform(data)

# 设置特征提取器
# fee = CSP()
fee = TS()

clf1 = make_pipeline(fee, svc, memory=memory)
scores = cross_val_score(clf1, data, label, cv=cv, n_jobs=15)
print("classifier1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

clf2 = make_pipeline(fee, lda, memory=memory)
scores = cross_val_score(clf2, data, label, cv=cv, n_jobs=15)
print("classifier2 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

clf3 = make_pipeline(fee, lr, memory=memory)
scores = cross_val_score(clf3, data, label, cv=cv, n_jobs=15)
print("classifier3 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

clf4 = make_pipeline(fee, knn, memory=memory)
scores = cross_val_score(clf4, data, label, cv=cv, n_jobs=15)
print("classifier4 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

clf5 = make_pipeline(fee, dtc, memory=memory)
scores = cross_val_score(clf5, data, label, cv=cv, n_jobs=15)
print("classifier5 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

clf6 = make_pipeline(fee, rfc, memory=memory)
scores = cross_val_score(clf6, data, label, cv=cv, n_jobs=15)
print("classifier6 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

clf7 = make_pipeline(fee, etc, memory=memory)
scores = cross_val_score(clf7, data, label, cv=cv, n_jobs=15)
print("classifier7 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

clf8 = make_pipeline(fee, abc, memory=memory)
scores = cross_val_score(clf8, data, label, cv=cv, n_jobs=15)
print("classifier8 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

clf9 = make_pipeline(fee, gbc, memory=memory)
scores = cross_val_score(clf9, data, label, cv=cv, n_jobs=15)
print("classifier9 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

clf10 = make_pipeline(fee, gnb, memory=memory)
scores = cross_val_score(clf10, data, label, cv=cv, n_jobs=15)
print("classifier10 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

clf11 = make_pipeline(fee, mlp, memory=memory)
scores = cross_val_score(clf11, data, label, cv=cv, n_jobs=15)
print("classifier12 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))