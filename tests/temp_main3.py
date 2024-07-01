


import numpy as np
from joblib import Memory

from pre_processing.preprocessing import Pre_Processing
from transfer_learning.tl_classifier import TL_Classifier

from loaddata import Dataset_Left_Right_MI


# 设置缓存目录
cachedir = '../my_cache_directory'
memory = Memory(cachedir, verbose=0)

preprocess = Pre_Processing(fs_new=160, fs_old=250, 
                       n_channels=np.arange(0, 28), 
                       start_time=0.5, end_time=3.5,
                       lowcut=None, highcut=None, )
Model = TL_Classifier(dpa_method='noTL', 
                      fee_method='CSP', 
                      fes_method='MIC-K', 
                      clf_method='SVM',
                      pre_est=preprocess.process,
                      memory=memory)

dataset_name = 'Pan2023'

fs = 250
freqband = [8,30]
datapath = r'E:\工作进展\小论文2023会议\数据处理python\datasets'

dataset = Dataset_Left_Right_MI(dataset_name,fs,fmin=freqband[0],fmax=freqband[1],tmin=0,tmax=4,path=datapath)
data, label = dataset.get_data([1])

Model.fit(data, label)
acc = Model.score(data, label)
print("Accuracy:", acc)