# 公共工具库
import os, time
import json
import numpy as np
import itertools
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
import time

# 私有工具库
from loaddata import Dataset_Left_Right_MI
from transfer_learning import TLSplitter, encode_datasets
from transfer_learning.algorithms import  Algorithms

# 设置参数
dataset_name = 'Pan2023'
fs = 250
freqband = [1,79]
datapath = r'E:\工作进展\小论文2023会议\数据处理python\datasets'

# 实例化数据集
dataset = Dataset_Left_Right_MI(dataset_name,fs,fmin=freqband[0],fmax=freqband[1],tmin=0,tmax=4,path=datapath)

# 设置交叉验证
cv = StratifiedShuffleSplit(n_splits=5, random_state=42) #可以控制训练集的数量
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_size = 0
print(f"Train size: {train_size}")

# 设置缓存目录
cachedir = '../my_cache_directory'

# 读取csv文件中的算法信息
# df = pd.read_csv('transfer_learning/algorithms_part.csv')
df = pd.read_csv('transfer_learning/algorithms_all.csv')

# 遍历每个subject
SN = []
for sub in dataset.subject_list:
    print(f"Subject {sub}...")
    
    # 加载数据
    data, label, info = dataset.get_data([sub])

    session_values = info['session'].unique()
    print('the session values are:', session_values)
    session_indices = info.groupby('session').apply(lambda x: x.index.tolist())

    # 将结果转换为字典，键为不同值，值为对应的索引列表
    session_index_dict = dict(zip(session_values, session_indices))

    Data, Label=[], []
    for session in session_values[:2]:
        Data.append(data[session_index_dict[session]])
        Label.append(label[session_index_dict[session]])

    X, y_enc, domain =encode_datasets(Data, Label)
    print(X.shape, y_enc.shape, len(domain))
    print(domain)

    target_domain = domain[-1]
    tl_cv = TLSplitter(target_domain=target_domain, cv=cv, no_calibration=False)
    if train_size == 0:
        tl_cv.no_calibration = True
    else:
        tl_cv.cv.train_size = train_size

    # 遍历csv文件，计算每个方法的准确率
    N = []
    for i in range(len(df)):
        # 读取当前行的key和value
        key = df.loc[i, 'Algorithms']
        value = df.loc[i, 'Algorithm_ID']
        value = eval(value)
        # 实例化模型
        Model = Algorithms(
        algorithm_id=value, 
        target_domain=target_domain, 
        memory_location=None,#=cachedir,
        fs_new=160, 
        fs_old=250, 
        n_channels=None, 
        start_time=0.5, 
        end_time=3.5, 
        lowcut=8, 
        highcut=30, 
        aug_method='time_window',
        window_width=None,
        window_step=0.5,
        pre_est=None,
        tl_mode='Calibration-free',
        )

        # # 读取当前行的准确率
        try:
            scores = cross_validate(Model, X, y_enc, cv=tl_cv, n_jobs=-1, return_train_score=True)
            train_time = scores['fit_time']
            train_score = scores['train_score']
            test_time = scores['score_time']
            test_score = scores['test_score']
            
            # 写入当前行的准确率
            df.loc[i, 'train_score'] = train_score.mean()
            df.loc[i, 'test_score'] = test_score.mean()
            df.loc[i, 'train_time'] = train_time.mean()
            df.loc[i, 'test_time'] = test_time.mean()

            print(f"{i+1}. Method {key} has an accuracy of {test_score.mean():.2f} +/- {test_score.std():.2f}")
            
            if (i+1) % 100 == 0:
                df.to_csv(f'results/algorithms_all_result_CF_{sub}.csv', index=False)
    
        except:
            N.append(i)
            print(f"{i+1}. Method {key} failed to run.")
        
        # ## 读取当前行的准确率
        # acc = []
        # for train, test in tl_cv.split(X, y_enc):
        #     X_train, y_train = X[train], y_enc[train]
        #     X_test, y_test = X[test], y_enc[test]
        #     Model.fit(X_train, y_train)
        #     score = Model.score(X_test, y_test)
        #     acc.append(score)
        # print(f"{i+1}. Method {key} has an accuracy of {np.mean(acc):.2f} +/- {np.std(acc):.2f}")

    SN.append(N)

# 输出各个sub的失败的行号
for i in range(len(SN)):
    print(f"Subject {i+1} failed at {SN[i]}")

