"""
Offline Main
Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/6/30
License: All rights reserved
"""

# 导入所需的包
import os, time
import pickle
import json
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from copy import deepcopy 
from neurodeckit.loaddata import Dataset_Left_Right_MI, Dataset_MI
from neurodeckit.transfer_learning import TLSplitter, encode_datasets
from neurodeckit.transfer_learning.algorithms import  Algorithms


def evaluate_algorithm(paraset, X, y_enc, tl_cv, target_domain):
    
    # 设置缓存目录
    cachedir = './my_cache_directory'
    
    # 实例化模型
    Model = Algorithms(
        algorithm_id=paraset['algorithm_id'], 
        target_domain=target_domain, 
        memory_location=cachedir,
        fs_new=None, 
        fs_old=None, 
        channels=paraset['channels'], 
        start_time=paraset['start_time'], 
        end_time=paraset['end_time'], 
        lowcut=paraset['lowcut'], 
        highcut=paraset['highcut'], 
        aug_method=paraset['aug_method'],
        window_width=None,
        window_step=0.5,
        pre_est=None,
        tl_mode=paraset['tl_mode'],
        cs_method=paraset['cs_method'],
        nelec=paraset['nelec'],
        domain_weight=paraset['domain_weight'],
    )
    
    Model = deepcopy(Model)

    try:
        scores = cross_validate(
            Model, X, y_enc, cv=tl_cv, n_jobs=1,#n_splits*n_repeats, 
            return_train_score=False)
        train_time = scores['fit_time'].mean()
        test_time = scores['score_time'].mean()
        test_score = scores['test_score'].mean()
    except:
        train_time = 0
        test_time = 0
        test_score = 0
    
    return {'train_time': train_time, 
            'test_time': test_time, 
            'test_score': test_score
            }

# 保存单个计算结果的函数
def save_result(paraset, result, filename):
    # 将结果转换为字典并保存为JSON格式
    result_dict = {
        'index':paraset['index'],
        'test_score': result['test_score'],
        'test_time': result['test_time'],
        'train_time': result['train_time'],
        'algorithm': paraset['algorithm'],
        'channels': paraset['channels'],
        'fs_new': None,
        'fs_old': None,
        'algorithm_id': paraset['algorithm_id'],
        'tl_mode': paraset['tl_mode'],
        'cs_method': paraset['cs_method'],
        'nelec': paraset['nelec'] ,
        'aug_method': paraset['aug_method'],
        'start_time': paraset['start_time'],
        'end_time': paraset['end_time'],
        'lowcut': paraset['lowcut'],
        'highcut': paraset['highcut'],
        'domain_weight': paraset['domain_weight'],
        }
    try:
        with gpu_lock:
            with open(filename, 'a') as f:
                json.dump(result_dict, f)
                f.write('\n')  # 换行，以便于读取时分割
    except:
        # print(f"Save result failed: {filename}")
        time.sleep(1) # 防止文件读写冲突
        with open(filename, 'a') as f:
            json.dump(result_dict, f)
            f.write('\n')  # 换行，以便于读取时分割

# 检查已完成的计算并返回未完成的计算列表
def check_completed_jobs(filename, parasets):
    if not os.path.exists(filename):
        return parasets
    else:
        with open(filename, 'r') as f:
            lines = f.readlines()
        completed_jobs = [json.loads(line.strip()) for line in lines]
        completed_jobs_ids = set([job['index'] for job in completed_jobs]) 
        uncompleted_jobs = [job for job in parasets if job['index'] not in completed_jobs_ids]
        return uncompleted_jobs

# 主函数
def main(X, y_enc, target_domain, domain_weight, parasets, tl_cv, filename):
    
    def process_run(paraset):
        paraset['domain_weight'] = domain_weight
        result = evaluate_algorithm(paraset, X, y_enc, tl_cv, target_domain)    
        save_result(paraset, result, filename)
    
    # 检查剩余的计算任务
    remaining_parasets = check_completed_jobs(filename, parasets)
    
    # 多进程计算
    with parallel_backend('loky', n_jobs=n_jobs):    
        try:
            Parallel(batch_size=1, verbose=len(parasets), timeout=3600)(
                delayed(process_run)(paraset) for paraset in remaining_parasets
                )
        except Exception as e:
            print(f"Timeout: {e}")
            pass
    
if __name__ == '__main__':
    
    #'Pan2023', 'BNCI2014_001', 'Shin2017A', 'BNCI2015_001', 'Lee2019_MI'
    dataset_name = 'Pan2023'
    
    # 定义结果保存目录
    folder = f'./results/cross_sessions/{dataset_name}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 定义GPU共享锁和值   
    manager = mp.Manager()
    gpu_lock = manager.Lock() 
    
    # 设置最大并行进程数
    cpu_jobs = 1
    n_jobs = min(int(mp.cpu_count()), cpu_jobs)
    
    # 设置数据集
    fs = 250
    datapath = r'E:\工作进展\小论文2023会议\数据处理python\datasets'
    # dataset = Dataset_MI(dataset_name,fs=fs,fmin=8,fmax=30,tmin=0,tmax=4,path=datapath)
    dataset = Dataset_Left_Right_MI(dataset_name,fs=fs,fmin=8,fmax=30,tmin=0,tmax=4,path=datapath)
    subjects = dataset.subject_list
    
    # 设置源域数据权重
    source_weight = 1

    # 设置交叉验证
    n_splits=1
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=2024) #可以控制训练集的数量
    train_size = 0

    # 读取参数设置
    with open('Parasets_BaseCF3177.pkl', 'rb') as f:   
        parasets = pickle.load(f)
    print(f"Parasets: {len(parasets)}")
    
    # 遍历每个subject
    for sub in subjects:
        print(f"Subject {sub}...")
        filename = folder + f'/result_sub{sub}.json'
        
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
        print(f"data shape: {X.shape}, label shape: {y_enc.shape}")
        print(f"All Domain: {domain}")

        target_domain = domain[-1]
        print(f"Target domain: {target_domain}")
        tl_cv = TLSplitter(target_domain=target_domain, cv=cv, no_calibration=False)
        if train_size == 0:
            tl_cv.no_calibration = True
        else:
            tl_cv.cv.train_size = train_size
              
        domain_weight = {d: 1 for d in domain}
        for d in domain:
            if d != target_domain:
                domain_weight[d] = source_weight
        print(f"Domain weight: {domain_weight}")

        # 运行主函数
        main(X, y_enc, target_domain, domain_weight, parasets, tl_cv, filename)



