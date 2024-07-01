"""
Offline Main
Author: Pan.LC <coreylin2023@outlook.com>
Date: 2024/6/30
License: All rights reserved
"""

# 公共工具库
import os, sys, time
import pickle
import json
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate

# 私有工具库
folder_path = 'MI_All_dev'
if folder_path not in sys.path:
    sys.path.append(folder_path)
    
from loaddata import LoadData
from transfer_learning import TLSplitter, encode_datasets
from transfer_learning.algorithms import  Algorithms


def evaluate_algorithm(paraset, X, y_enc, tl_cv, target_domain):
     
    # 实例化模型
    Model = Algorithms(
        algorithm_id=paraset['algorithm_id'], 
        target_domain=target_domain, 
        memory_location=cachedir,
        fs_new=fs_new, 
        fs_old=fs_old, 
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
    )

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
    
    return {'train_time': train_time, 'test_time': test_time, 'test_score': test_score}

# 保存单个计算结果的函数
def save_result(paraset, result, filename):
    # 将结果转换为字典并保存为JSON格式
    result_dict = {
        'index':paraset['index'],
        'test_score': result['test_score'],
        'train_time': result['train_time'],
        'test_time': result['test_time'],
        'algorithm': paraset['algorithm'],
        'channels': paraset['channels'] if isinstance(paraset['channels'], list) else paraset['channels'].tolist(),
        'fs_new': fs_new,
        'fs_old': fs_old,
        'algorithm_id': paraset['algorithm_id'],
        'tl_mode': paraset['tl_mode'],
        'cs_method': paraset['cs_method'],
        'nelec': paraset['nelec'] ,
        'aug_method': paraset['aug_method'],
        'start_time': paraset['start_time'],
        'end_time': paraset['end_time'],
        'lowcut': paraset['lowcut'],
        'highcut': paraset['highcut'],
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
def main(X, y_enc, target_domain, parasets, tl_cv, filename):
    
    def process_run(paraset):
        result = evaluate_algorithm(paraset, X, y_enc, tl_cv, target_domain)    
        save_result(paraset, result, filename)
    
    # 检查剩余的计算任务
    remaining_parasets = check_completed_jobs(filename, parasets)
    
    # 多进程计算
    with parallel_backend('loky', n_jobs=n_jobs):    
        Parallel(batch_size=1, verbose=len(parasets))(
            delayed(process_run)(paraset) for paraset in remaining_parasets
            )
    
if __name__ == '__main__':
    
    # 定义GPU共享锁和值   
    manager = mp.Manager()
    gpu_lock = manager.Lock() 
    
    # 设置缓存目录
    cachedir = '../my_cache_directory'
    
    # 设置最大并行进程数
    cpu_jobs = 60
    n_jobs = min(int(mp.cpu_count()), cpu_jobs)
    
    # 设置数据集
    fs_old = 250
    fs_new = 160
    dataA = LoadData('TrainData/A', fs=fs_old)
    subjects = dataA.subject_list
    datas, labels = dataA.get_data(subjects)

    # 设置交叉验证
    n_splits=3
    n_repeats=1
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2024)

    # 读取参数设置
    with open('Parasets9000.pkl', 'rb') as f:   
        parasets = pickle.load(f)
    
    # 遍历每个subject
    for sub in subjects:
        print(f"Subject {sub}...")
        filename = f'Results/result_sub{sub}.json'
        
        X, y_enc, domain =encode_datasets(datas, labels)
        print(f"data shape: {X.shape}, label shape: {y_enc.shape}")
        print(f"All Domain: {domain}")

        target_domain = f'S{sub}'
        print(f"Target domain: {target_domain}")
        
        tl_cv = TLSplitter(target_domain=target_domain, cv=cv, no_calibration=False)

        # 运行主函数
        main(X, y_enc, target_domain, parasets, tl_cv, filename)



