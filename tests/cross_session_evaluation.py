# RAVEplus
# Authors: Pan.LC
# Date: 2024/6/24

# 公共工具库
import os, time
import json
import numpy as np
import itertools
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import RepeatedStratifiedKFold
from contextlib import redirect_stdout, redirect_stderr
import torch

# 私有工具库
from loaddata import Dataset_Left_Right_MI
from deep_learning.dl_classifier import DL_Classifier
from pre_processing.preprocessing import Pre_Processing
from transfer_learning.tl_classifier import TL_Classifier
from transfer_learning import TLSplitter, encode_datasets




# 设置参数
dataset_name = 'Pan2023'
fs = 250
freqband = [8,30]
datapath = r'E:\工作进展\小论文2023会议\数据处理python\datasets'

# 加载数据
dataset = Dataset_Left_Right_MI(dataset_name,fs,fmin=freqband[0],fmax=freqband[1],tmin=0,tmax=4,path=datapath)

sdata, slabel = [], []
for i in dataset.subject_list:    
    data, label = dataset.get_data([i])
    sdata.append(data)
    slabel.append(label)
