# -*- coding: utf-8 -*-

"""
    
   This is a script for formating the EEG data for deep learning models.
   
   The script includes:
   1. FilterBank: A class for filtering the EEG data using Chebyshev type 2 filter.
   2. Formatdata: A class for formating the EEG data for deep learning models.
   
   The script is modified to include the following methods:
   1. _generate_rsf_filter: A method for generating the RSF filter for EEG data preprocessing.
   2. _apply_rsf_filter: A method for applying the RSF filter to EEG data.
   3. fit_transform: A method for fitting and transforming the EEG data.
   4. transform: A method for transforming the EEG data.
   
   The script is modified to include the following parameters:
   1. sample_length: The length of the EEG data in seconds.
   2. fs: The sampling frequency of the EEG data.
   3. alg_name: The name of the deep learning algorithm.
   4. rsf_method: The method for RSF filter.
   5. rsf_dim: The dimension of the RSF filter.
   6. freq_seg: The number of frequency segments for RSF filter.
   7. is_training: A flag for indicating whether the data is for training or testing.
   
   Author: LC.Pan
   Date: 2024/3/13
   License: BSD 3-Clause License
   
"""


import numpy as np
import torch
from torch.autograd import Variable
import scipy.signal as signal
from scipy.signal import cheb2ord, butter, filtfilt
from scipy.linalg import eigvalsh
from pyriemann.estimation import Covariances
from sklearn.base import BaseEstimator, TransformerMixin
from ..pre_processing import RSF
from ..utils import generate_intervals


class FilterBank:
    def __init__(self, fs, freqbands = generate_intervals(4, 4, (4, 40))):
        self.fs           = fs
        self.freqbands    = freqbands

        self.f_trans      = 2
        self.gpass        = 3
        self.gstop        = 30
        self.filter_coeff = {}
        self.n_bands      = len(self.freqbands)

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2
 
        for i, f_pass in enumerate(self.freqbands):
            f_stop    = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp        = tuple(f/Nyquist_freq for f in f_pass)
            ws        = tuple(f/Nyquist_freq for f in f_stop)
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a      = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i:{'b':b,'a':a}})
            
        return self.filter_coeff
    
    def filter_data(self,eeg_data,window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape

        if window_details:
            n_samples = int(self.fs * (window_details.get('tmax') - window_details.get('tmin')))
            #+1

        filtered_data = np.zeros((self.n_bands, n_trials, n_channels, n_samples))

        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            # 使用filtfilt函数代替lfilter
            for j in range(n_trials):
                eeg_data_filtered = signal.filtfilt(b, a, eeg_data[j, :, :], axis=1)
                if window_details:
                    start = int(window_details.get('tmin') * self.fs)
                    end = int(window_details.get('tmax') * self.fs)
                    eeg_data_filtered = eeg_data_filtered[:, start:end]
                filtered_data[i, j, :, :] = eeg_data_filtered

        return filtered_data


class Formatdata(BaseEstimator, TransformerMixin):
    def __init__(self, fs, n_times = None, alg_name ='Tensor_CSPNet', rsf_method='none', rsf_dim=8, freqbands=None, **kwargs):
        self.fs = fs
        self.alg_name = alg_name
        self.rsf_method = rsf_method if rsf_method is not None else 'none'
        self.rsf_dim = rsf_dim
        self.freq_seg = 4
        self.is_training = False
        self.freqbands = generate_intervals(4, 4, (4, 40)) if freqbands is None else freqbands
        self.dtype = kwargs.get('dtype', 'float64')
        self.n_bands = 1
        
        if self.alg_name in ['Tensor_CSPNet','oTensor_CSPNet']:
            self.time_seg = self._calculate_time_segments(n_times, fs)

        elif self.alg_name in ['Graph_CSPNet','oGraph_CSPNet']:
            self.time_freq_graph = self._generate_time_freq_graph(n_times, fs)
            self.block_dims = [
                          len(self.time_freq_graph['1']), 
                          len(self.time_freq_graph['2']), 
                          len(self.time_freq_graph['3']) + len(self.time_freq_graph['4']) + len(self.time_freq_graph['5']) + len(self.time_freq_graph['6']), 
                          len(self.time_freq_graph['7']) + len(self.time_freq_graph['8']) + len(self.time_freq_graph['9'])
                          ]
            m = len(self.time_freq_graph['1'])
            self.time_windows = [m, m, m, m*2]
            self.graph_M = None
    
    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):   
        # 设计巴特沃斯带通滤波器
        nyq = 0.5 * fs # 奈奎斯特频率
        low = lowcut / nyq # 归一化低截止频率
        high = highcut / nyq # 归一化高截止频率
        b, a = butter(order, [low, high], btype='band') # 返回滤波器的分子和分母系数

        # 对data进行滤波
        data_filtered = filtfilt(b, a, data) # 使用filtfilt函数可以避免相位延迟

        return data_filtered
            
    def _generate_time_freq_graph(self, signal_length, sampling_rate):
        
        # 初始化time_freq_graph字典
        time_freq_graph = {}

        # 计算前6个键基于采样率的部分数
        parts_for_first_six = signal_length // (sampling_rate * 1)  # 1秒的部分
        # 计算后3个键基于采样率的部分数
        parts_for_last_three = signal_length // (sampling_rate // 2)  # 0.5秒的部分

        # 为前6个键生成值
        for i in range(1, 7):
            time_freq_graph[str(i)] = [[j * sampling_rate, (j + 1) * sampling_rate] for j in range(parts_for_first_six)]

        # 为后3个键生成值
        for i in range(7, 10):
            time_freq_graph[str(i)] = [[j * (sampling_rate // 2), (j + 1) * (sampling_rate // 2)] for j in range(parts_for_last_three)]

        return time_freq_graph
    
    def _calculate_time_segments(self, sample_length, sampling_rate, window_duration=1, step_duration=1):
        """
        根据给定的样本长度、采样率、窗宽和步长，计算时间窗范围列表。

        参数:
        sample_length (int): 样本的总长度。
        sampling_rate (int): 采样率，单位为Hz。
        window_duration (float): 窗宽，单位为秒。
        step_duration (float): 步长，单位为秒。

        返回:
        list: 包含时间窗范围的列表，每个时间窗范围是一个[start, end]列表。
        """
        # 计算窗宽和步长对应的样本点数
        window_size = int(window_duration * sampling_rate)
        step_size = int(step_duration * sampling_rate)

        # 初始化时间窗范围列表
        time_segments = []

        # 计算时间窗范围
        for start_point in range(0, sample_length - window_size + 1, step_size):
            end_point = start_point + window_size
            time_segments.append([start_point, end_point])

        return time_segments
    
    def _tensor_stack(self, X):

        if self.alg_name in ['Tensor_CSPNet','oTensor_CSPNet']:
            '''
            For Tensor-CSPNet:

            Step 1: Segment the signal on each given time intervals.
                    e.g., (trials, frequency bands, channels, timestamps) ---> 
                    (trials, temporal segments, frequency bands, channels, timestamps);
            Step 2: Take covariance.
                    e.g., (trials, temporal segments, frequency bands, channels, timestamps) --->
                    (trials, temporal segments, frequency bands, channels, channels).
            '''
            temporal_seg   = []
            for [a, b] in self.time_seg:
                temporal_seg.append(np.expand_dims(X[:, :, :, a:b], axis = 1))
            temporal_seg   = np.concatenate(temporal_seg, axis = 1)

            stack_tensor   = []
            for i in range(temporal_seg.shape[0]):
                cov_stack  = []
                for j in range(temporal_seg.shape[1]):
                    #cov_stack.append(Covariances().transform(temporal_seg[i, j]))
                    #We apply the shrinkage regularization on input SPD manifolds.
                    # cov_stack.append(Shrinkage(1e-2).transform(Covariances().transform(temporal_seg[i, j])))
                    cov_stack.append(Covariances(estimator='lwf').transform(temporal_seg[i, j]))
                stack_tensor.append(np.stack(cov_stack, axis = 0))
            stack_tensor   = np.stack(stack_tensor, axis = 0)

        elif self.alg_name in ['Graph_CSPNet','oGraph_CSPNet']:
            '''
            For Graph-CSPNet:
            Take covariance on each temporal intervals given in the time-frequency graph. 
            e.g., (trials, frequency bands, channels, timestamps) ---> 
                  (trials, segment, channels, channels);     
            '''
            stack_tensor   = []
            for i in range(1, X.shape[1]+1):
                for [a, b] in self.time_freq_graph[str(i)]:
                    cov_record = []
                    for j in range(X.shape[0]):
                    #cov_record.append(Covariances().transform(x_fb[j, i-1:i, :, a:b]))
                    #We apply the shrinkage regularization on input SPD manifolds.
                        cov_record.append(Covariances(estimator='lwf').transform(X[j, i-1:i, :, a:b]))
                    stack_tensor.append(np.expand_dims(np.concatenate(cov_record, axis = 0), axis = 1))
            stack_tensor   = np.concatenate(stack_tensor, axis = 1)

        return stack_tensor
    
    def _riemann_distance(self, A, B):
        #geodesic distance under metric AIRM. 
        return np.sqrt((np.log(eigvalsh(A, B))**2).sum())
        
    def LGT_graph_matrix_fn(self, gamma = 50, time_step = [2, 2, 2, 4], freq_step = [1, 1, 4, 3]):
        '''
        time_step: a list, step of diffusion to right direction.
        freq_step: a list, step of diffusion to down direction.
        gamma: Gaussian coefficent.
        '''
        # 校正time_step和freq_step以确保它们不超过self.time_windows的边界
        time_step = [min(ts, tw) for ts, tw in zip(time_step, self.time_windows)]
        freq_step = [min(fs, tw) for fs, tw in zip(freq_step, self.time_windows)]
    
        A = np.zeros((sum(self.block_dims), sum(self.block_dims))) + np.eye(sum(self.block_dims))
        start_point = 0
        for m in range(len(self.block_dims)):
            for i in range(self.block_dims[m]):
                max_time_step = min(self.time_windows[m] - 1 - (i % self.time_windows[m]), time_step[m])
                for j in range(i + 1, i + max_time_step + 1):
                    A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                    A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
                for freq_mul in range(1, freq_step[m]+1):
                    for j in range(i+ freq_mul*self.time_windows[m], i + freq_mul*self.time_windows[m] + max_time_step + 1):
                        if j < self.block_dims[m]: 
                            A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                            A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
            start_point += self.block_dims[m]

        D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis = 0))))

        return np.matmul(D, A), A
            
    def fit(self, X=None, y=None):
        
        self.labels = y
        self.is_training = True
        
        return self
                  
    def transform(self, X):
        
        # 沿用原作的Graph_CSPNet的频段设置
        if self.alg_name in ['Graph_CSPNet', 'oGraph_CSPNet']:
            self.freqbands = generate_intervals(4, 4, (4, 40))
        
        # 滤波
        fbank = None
        if self.alg_name in ['Tensor_CSPNet', 'Graph_CSPNet']:
            fbank = FilterBank(fs = self.fs, freqbands = self.freqbands)
            _     = fbank.get_filter_coeff()
            '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
            X_fb  = fbank.filter_data(X).transpose(1, 0, 2, 3)
            if self.rsf_method != 'none':
                if self.is_training:
                    self.rsf_transformer = RSF(self.rsf_dim, self.rsf_method)
                    X_fb = np.stack([self.rsf_transformer.fit_transform(X_fb[:, band, :, :], self.labels)
                                    for band in range(X_fb.shape[1])], axis=1)
                else: 
                    X_fb = np.stack([self.rsf_transformer.transform(X_fb[:, band, :, :])
                                    for band in range(X_fb.shape[1])], axis=1)
            X_transformed = self._tensor_stack(X_fb)
        
        elif self.alg_name in ['oTensor_CSPNet', 'oGraph_CSPNet']:
            fbank = FilterBank(fs = self.fs, freqbands = self.freqbands)
            _     = fbank.get_filter_coeff()
            '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
            X_fb  = fbank.filter_data(X).transpose(1, 0, 2, 3)
            X_transformed = self._tensor_stack(X_fb)
            
        elif self.alg_name in ['FBCNet']:
            fbank = FilterBank(fs = self.fs, freqbands = self.freqbands)
            _     = fbank.get_filter_coeff()
            '''The shape of x_fb is No. of (trials, channels, timestamps, frequency bands)'''
            X_fb  = fbank.filter_data(X).transpose(1, 2, 3, 0)                
            if self.rsf_method != 'none':
                if self.is_training:
                    self.rsf_transformer = RSF(self.rsf_dim, self.rsf_method)
                    X_fb = np.stack([self.rsf_transformer.fit_transform(X_fb[:, :, :, band], self.labels)
                                    for band in range(X_fb.shape[3])], axis=3)
                else:
                    X_fb = np.stack([self.rsf_transformer.transform(X_fb[:, :, :, band])
                                    for band in range(X_fb.shape[3])], axis=3)
            X_transformed = np.expand_dims(X_fb, axis=1) # trials, 1, channels, timestamps, frequency bands
        
        elif self.alg_name in ['oFBCNet']:
            fbank = FilterBank(fs = self.fs, freqbands = self.freqbands)
            _     = fbank.get_filter_coeff()
            '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
            X_fb  = fbank.filter_data(X).transpose(1, 2, 3, 0)  
            X_transformed = np.expand_dims(X_fb, axis=1) # trials, 1, channels, timestamps, frequency bands
        
        elif self.alg_name in ['IFNet']:
            fbank = FilterBank(fs = self.fs, freqbands = [(4, 16), (16, 40)])
            _     = fbank.get_filter_coeff()
            X_fb  = fbank.filter_data(X).transpose(1, 0, 2, 3)  
            X_transformed = X_fb.reshape(X_fb.shape[0], X_fb.shape[1] * X_fb.shape[2], X_fb.shape[3]) # trials, channels*frequency bands, timestamps
        
        elif self.alg_name in ['LightConvNet']:
            fbank = FilterBank(fs = self.fs, freqbands = self.freqbands)
            _     = fbank.get_filter_coeff()
            '''The shape of x_fb is No. of (trials, channels, timestamps, frequency bands)'''
            X_fb  = fbank.filter_data(X).transpose(1, 2, 3, 0)                
            if self.rsf_method != 'none':
                if self.is_training:
                    self.rsf_transformer = RSF(self.rsf_dim, self.rsf_method)
                    X_fb = np.stack([self.rsf_transformer.fit_transform(X_fb[:, :, :, band], self.labels)
                                    for band in range(X_fb.shape[3])], axis=3)
                else:
                    X_fb = np.stack([self.rsf_transformer.transform(X_fb[:, :, :, band])
                                    for band in range(X_fb.shape[3])], axis=3)
            X_transformed = X_fb.transpose(0, 3, 1, 2) # trials, frequency bands, channels, timestamps
        
        elif self.alg_name in ['oLightConvNet']:
            fbank = FilterBank(fs = self.fs, freqbands = self.freqbands)
            _     = fbank.get_filter_coeff()
            '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
            X_fb  = fbank.filter_data(X).transpose(1, 2, 3, 0)  
            X_transformed = X_fb.transpose(0, 3, 1, 2) # trials, frequency bands, channels, timestamps
        
        elif self.alg_name in ['FB-CSP','FB-CSP-LDA','FB-CSP-SVM','RSF-FB-CSP-LDA','RSF-FB-CSP-SVM']:
            fbank = FilterBank(fs = self.fs, freqbands = self.freqbands)
            _     = fbank.get_filter_coeff()
            '''The shape of x_fb is No. of (trials, channels, timestamps, frequency bands)'''
            X_fb  = fbank.filter_data(X).transpose(1, 2, 3, 0)   
            if self.rsf_method != 'none':
                if self.is_training:
                    self.rsf_transformer = RSF(self.rsf_dim, self.rsf_method)
                    X_fb = np.stack([self.rsf_transformer.fit_transform(X_fb[:, :, :, band], self.labels)
                                    for band in range(X_fb.shape[3])], axis=3)
                else:
                    X_fb = np.stack([self.rsf_transformer.transform(X_fb[:, :, :, band])
                                    for band in range(X_fb.shape[3])], axis=3)
            X_transformed = X_fb
        
        # elif self.alg_name in ['EEGNet', 'ShallowNet', 'DeepNet', 'LMDANet']:
        else:
            if self.freqbands is not None and len(self.freqbands) == 2:
                X_fb  = self._butter_bandpass_filter(X, self.freqbands[0], self.freqbands[1], self.fs) 
            else:
                X_fb  = X
            
            if self.rsf_method != 'none':
                if self.is_training:
                    self.rsf_transformer = RSF(self.rsf_dim, self.rsf_method)
                    X_fb = self.rsf_transformer.fit_transform(X_fb, self.labels)
                else: 
                    X_fb = self.rsf_transformer.transform(X_fb)
            X_transformed = X_fb
        
        
        if fbank is not None:
            self.n_bands = fbank.n_bands
        
        # 对于Graph-CSPNet，需要返回LGT的图矩阵 
        if self.is_training and self.alg_name in ['Graph_CSPNet', 'oGraph_CSPNet']:
            self.lattice = np.mean(X_transformed, axis = 0)
            self.graph_M, _ = self.LGT_graph_matrix_fn()
            if self.dtype == 'float64':
                self.graph_M = Variable(torch.from_numpy(self.graph_M)).double()
            elif self.dtype == 'float32':
                self.graph_M = Variable(torch.from_numpy(self.graph_M)).float()
            else:
                raise ValueError('dtype should be float32 or float64.')
        
        self.is_training = False
        return X_transformed.astype(self.dtype)
        