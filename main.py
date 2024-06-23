import numpy as np
from joblib import Memory
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from pre_processing.preprocessing import Pre_Processing
from transfer_learning.tl_classifier import TL_Classifier
from transfer_learning import TLSplitter, encode_datasets
from loaddata import Dataset_Left_Right_MI

# 设置参数
dataset_name = 'Pan2023'
fs = 250
freqband = [8,30]
datapath = r'E:\工作进展\小论文2023会议\数据处理python\datasets'

# 加载数据
dataset = Dataset_Left_Right_MI(dataset_name,fs,fmin=freqband[0],fmax=freqband[1],tmin=0,tmax=4,path=datapath)
sdata, slabel = [], []
for i in range(1,4):    
    data, label = dataset.get_data([i])
    sdata.append(data)
    slabel.append(label)
    
X, y_enc, domain =encode_datasets(sdata, slabel)
print(X.shape, y_enc.shape, len(domain))
print(domain)

# 设置缓存目录
cachedir = '../my_cache_directory'
memory = Memory(cachedir, verbose=0)

# 实例化模型
preprocess = Pre_Processing(fs_new=160, fs_old=250, 
                       n_channels=np.arange(0, 28), 
                       start_time=0.5, end_time=3.5,
                       lowcut=None, highcut=None, )
Model = TL_Classifier(dpa_method='EA', 
                      fee_method='CSP', 
                      fes_method='MIC-K', 
                      clf_method='SVM',
                      pre_est=preprocess.process,
                      memory=memory,
                      target_domain=domain[0],
                      )

# 交叉验证
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
tl_cv = TLSplitter(target_domain=domain[0], cv=cv)

#%%
# acc = []
# for train, test in tl_cv.split(X, y_enc):
#     X_train, y_train = X[train], y_enc[train]
#     X_test, y_test = X[test], y_enc[test]
#     Model.fit(X_train, y_train)
#     score = Model.score(X_test, y_test)
#     acc.append(score)
#     print("Score: %0.2f" % score)
# print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(acc), np.std(acc)))

#%%
scores = cross_val_score(Model, X, y_enc, cv=tl_cv, n_jobs=15)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

# %%
