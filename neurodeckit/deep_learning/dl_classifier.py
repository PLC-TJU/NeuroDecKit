"""
DeepL_Classifier: Deep Learning Classifier for EEG Data
Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2024/3/14
License: All rights reserved

Introduction:
The DeepL_Classifier is a Python class designed to facilitate the training and evaluation of various deep 
learning models for electroencephalogram (EEG) data classification. It is built on top of PyTorch and 
scikit-learn, providing a flexible and easy-to-use interface for experimenting with different neural 
network architectures.

Features:
    1> Supports multiple EEG-based deep learning models.
    2> Integrates with scikit-learn's BaseEstimator and TransformerMixin for compatibility with scikit-learn workflows.
    3> Allows customization of training parameters such as batch size, learning rate, and number of epochs.
    4> Can be used with any device that PyTorch supports (CPU or CUDA-enabled GPU).

Usage:
To use the DeepL_Classifier, you need to initialize it with the desired model name and training parameters. 
Then, you can fit the model to your training data and use it to transform (predict) on new data.

Initialization Parameters:
    1> model_name (str): Name of the deep learning model to use. Supported models include: 
        'ShallowNet', 'ShallowFBCSPNet', 'DeepNet', 'Deep4Net' 
                R. T. Schirrmeister et al., "Deep learning with convolutional neural networks for EEG decoding and 
                visualization," Hum Brain Mapp, vol. 38, no. 11, pp. 5391-5420, Nov 2017, doi: 10.1002/hbm.23730.
        'EEGNet','EEGNetv4'      
                V. J. Lawhern et al., "EEGNet: a compact convolutional neural network for EEG-based brain-computer
                interfaces," J Neural Eng, vol. 15, no. 5, p. 056013, Oct 2018, doi: 10.1088/1741-2552/aace8c.
        'FBCNet'
                R. Mane et al., "FBCNet: A multi-view convolutional neural network for brain-computer interface," 
                arXiv preprint, vol. 2104.01233, Mar 2021. [Online]. Available: https://arxiv.org/abs/2104.01233.
        'Tensor_CSPNet'
                C. Ju and C. Guan, "Tensor-CSPNet: A Novel Geometric Deep Learning Framework for Motor Imagery 
                Classification," IEEE Trans Neural Netw Learn Syst, vol. 34, no. 12, pp. 10955-10969, Dec 2023, 
                doi: 10.1109/TNNLS.2022.3172108.
        'Graph_CSPNet'
                C. Ju and C. Guan, "Graph Neural Networks on SPD Manifolds for Motor Imagery Classification: A 
                Perspective From the Time-Frequency Analysis," IEEE Trans Neural Netw Learn Syst, vol. PP, pp. 1-15, 
                Sep 19 2023, doi: 10.1109/TNNLS.2023.3307470.
        'LMDANet'
                Z. Miao, M. Zhao, X. Zhang, and D. Ming, "LMDA-Net:A lightweight multi-dimensional attention network
                for general EEG-based brain-computer interfaces and interpretability," Neuroimage, vol. 276, p. 120209,
                Aug 1 2023, doi: 10.1016/j.neuroimage.2023.120209.
        'EEGConformer'
                Y. Song, Q. Zheng, B. Liu, and X. Gao, “EEG conformer: Convolutional transformer for EEG decoding and 
                visualization,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, 
                vol. 31, pp. 710–719, 2022.
        'LightConvNet'
                X. Ma, W. Chen, Z. Pei, J. Liu, B. Huang, and J. Chen, “A Temporal Dependency Learning CNN With Attention 
                Mechanism for MI-EEG Decoding,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, 
                vol. 31, pp. 3188–3200, 2023, doi: 10.1109/TNSRE.2023.3299355.
        'IFNet'
                J. Wang, L. Yao, and Y. Wang, “IFNet: An Interactive Frequency Convolutional Neural Network for 
                Enhancing Motor Imagery Decoding From EEG,” IEEE Transactions on Neural Systems and Rehabilitation 
                Engineering, vol. 31, pp. 1900–1911, 2023, doi: 10.1109/TNSRE.2023.3257319.
        'MSVTNet'
                K. Liu et al., "MSVTNet: Multi-Scale Vision Transformer Neural Network for EEG-Based Motor Imagery 
                Decoding," IEEE Journal of Biomedical and Health Informatics, vol. 28, no. 12, pp. 7126–7137, 
                Dec. 2024, doi: 10.1109/JBHI.2024.3450753.
        

             
    2> fs (int): Sampling frequency of the EEG data.
    3> batch_size (int): Number of samples per batch during training.
    4> lr (float): Learning rate for the optimizer.
    5> max_epochs (int): Maximum number of epochs for training.
    6> device (str): Device to run the computations on ('cpu' or 'cuda').
    7> **kwargs: Additional keyword arguments to pass to the underlying neural network.

Methods:
fit(X, y): Trains the model on the provided data.
    X (array-like): Training data with shape (n_samples, n_channels, n_times).
    y (array-like): Target labels with shape (n_samples,).
    returns:
        self: The fitted model.

transform(X): Transforms on the provided data.
    X (array-like): Data to transform with shape (n_samples, n_channels, n_times).
    returns:
        features (array-like): Transformed features with shape (n_samples, n_features).

predict(X): Predicts labels for the given data.
    X (array-like): Data to predict with shape (n_samples, n_channels, n_times).
    returns:
        predictions (array-like): Predicted labels with shape (n_samples,).

score(X, y): Computes the accuracy of the model on the given data.
    X (array-like): Data to predict with shape (n_samples, n_channels, n_times).
    y (array-like): True labels for computing accuracy.
    returns:
        accuracy (float): Accuracy of the model on the given data.

Input/Output Details:
    1> Input data (X) should be a 3D NumPy array or any array-like structure compatible with PyTorch, 
       with dimensions corresponding to (samples, channels, time points).
    2> Output predictions are NumPy arrays containing the predicted labels for each sample.
    3> If true labels (y) are provided during the transform, the method also returns the accuracy as a float.
    
Please ensure that the input data is preprocessed and compatible with the model requirements. 
The sampling frequency (fs) should match the frequency used during data collection.

This documentation provides an overview of the DL_Classifier class, its methods, and how to use it for EEG data 
classification tasks. For more detailed information on the individual models and their specific requirements, 
refer to the respective model documentation.

Example:
from dl_classifier import DL_Classifier

# Initialize the classifier with EEGNet model and training parameters
classifier = DeepL_Classifier(model_name='EEGNet', fs=128, batch_size=32, lr=1e-2, max_epochs=200, device='cpu')

# Fit the classifier to the training data
classifier.fit(train_data, train_labels)

# Transform (predict) on the test data
test_features = classifier.transform(test_data)

# Predict labels for the test data
test_predictions = classifier.predict(test_data)

# Compute accuracy on the test data
test_accuracy = classifier.score(test_data, test_labels)

"""

import torch
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from . import Formatdata
from . import (EEGNet, ShallowNet, DeepNet,         
               EEGNetv4, ShallowFBCSPNet, Deep4Net,  #推荐使用这三种（原版）
               FBCNet, Tensor_CSPNet, Graph_CSPNet, 
               LMDANet, MSVTNet, IFNet, EEGConformer, LightConvNet
               )



def check_nn(est):
    """Check if a given estimator is valid."""

    # Check estimator exist and return the correct function
    estimators = {
        'EEGNet': EEGNet,
        'EEGNetv4': EEGNetv4,
        'ShallowNet': ShallowNet,
        'ShallowFBCSPNet': ShallowFBCSPNet,
        'DeepNet': DeepNet,
        'Deep4Net': Deep4Net,  
        'FBCNet': FBCNet, #oFBCNet与FBCNet的区别在于当rsf_method非none时，oFBCNet使用的是原始的FBCNet，否则使用对各个子频带的数据进行额外的rsf空间滤波
        'oFBCNet': FBCNet,
        'Tensor_CSPNet': Tensor_CSPNet, # 同理
        'oTensor_CSPNet': Tensor_CSPNet,
        'Graph_CSPNet': Graph_CSPNet, # 同理
        'oGraph_CSPNet': Graph_CSPNet,
        'LMDANet': LMDANet,
        'MSVTNet': MSVTNet,
        'IFNet': IFNet,
        'EEGConformer': EEGConformer,
        'LightConvNet': LightConvNet,
    }

    if callable(est):
        # All good (cross your fingers)
        pass
    elif est in estimators.keys():
        # Map the corresponding estimator
        est = estimators[est]
    else:
        # raise an error
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
            callable function""" % (est, (' , ').join(estimators.keys())))
    return est


class DL_Classifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, 
                 model_name='EEGNet', 
                 n_classes=2, 
                 fs=128, 
                 batch_size=32, 
                 lr=1e-3, 
                 max_epochs=300, 
                 device='cpu', 
                 freqbands=None, 
                 dtype='float32', # 默认使用float32以节省内存, 也可以使用float64
                 seed=42, 
                 patience=50, 
                 rsf_method='none', 
                 rsf_dim=10, 
                 **kwargs
                 ):

        self.model_name = model_name
        self.n_classes = n_classes
        self.fs = fs
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.device = 'cuda' if device == 'gpu' else device
        self.pp_name = None
        self.nn_name = None
        self.Process = None
        self.Net = None
        self.Model = None
        self.rsf_method = rsf_method
        self.rsf_dim = rsf_dim
        self.freqbands = freqbands
        self.dtype = dtype 
        self.seed = seed
        self.patience = patience
        self.kwargs = kwargs
        
        self.n_bands = 1
        
        self.net_params = {
            'batch_size': self.batch_size,
            'lr': self.lr,
            'max_epochs': self.max_epochs,
            'device': self.device,
            'dtype': self.dtype,
            'seed': self.seed,
            'patience': self.patience,
            **kwargs  # This allows for any additional parameters to be passed
        }

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        self.n_classes = len(self.classes_)
        
        # 复制数据
        X = X.copy()
        n_channels, n_samples = X.shape[1], X.shape[2]
        
        # 提取实际的模型名称
        self.pp_name = self.model_name.split('-')[0] if '-' in self.model_name else 'None'
        self.nn_name = self.model_name.split('-')[-1] if '-' in self.model_name else self.model_name
        
        # 转换数据
        Process = None
        if self.pp_name.lower() == 'rsf' or self.nn_name in ['Graph_CSPNet', 'Tensor_CSPNet', 'FBCNet',
                                                             'oGraph_CSPNet', 'oTensor_CSPNet', 'oFBCNet',
                                                             'LightConvNet', 'IFNet'
                                                             ]:
            Process = Formatdata(fs=self.fs, n_times=X.shape[2], alg_name=self.nn_name, dtype=self.dtype,   
                                 rsf_method=self.rsf_method, rsf_dim=self.rsf_dim, freqbands=self.freqbands)
            X = Process.fit_transform(X, y)
            self.n_bands = Process.n_bands
        
        # 实例化深度学习模型
        Network = check_nn(self.nn_name)
        if self.nn_name in ['Graph_CSPNet', 'oGraph_CSPNet']:
            graph_M = Process.graph_M.to(self.device)
            Net = Network(graph_M, X.shape[1], X.shape[2], n_classes = self.n_classes, net_params=self.net_params)
        elif self.nn_name in ['Tensor_CSPNet', 'oTensor_CSPNet']:
            Net = Network(len(Process.time_seg), X.shape[1] * X.shape[2], X.shape[3], n_classes = self.n_classes, 
                          n_bands=self.n_bands, net_params=self.net_params)
        elif self.nn_name in ['FBCNet', 'oFBCNet']:
            Net = Network(X.shape[2], X.shape[3], self.n_classes, n_bands=self.n_bands, net_params=self.net_params)
        elif self.nn_name in ['LightConvNet']:
            Net = Network(X.shape[2], X.shape[3], self.n_classes, n_bands=self.n_bands, win_len=self.fs, net_params=self.net_params)# 默认1秒时间窗宽
        elif self.nn_name in ['IFNet']:
            Net = Network(n_channels, n_samples, self.n_classes, net_params=self.net_params)
        else:
            Net = Network(X.shape[1], X.shape[2], self.n_classes, net_params=self.net_params)
        
        # 训练深度学习模型
        Net.fit(X.astype(self.dtype).copy(), y.astype('int64').copy())
        
        # 保存模型和预处理器
        self.Process = Process
        self.Net = Net
        self.Model = make_pipeline(Process, Net) if Process is not None else Net
        
        return self

    def predict(self, X):
        # 确保模型已经训练
        if hasattr(self, 'Model'):
            predictions = self.Model.predict(X.astype(self.dtype).copy())
        else:
            raise ValueError("Model is not trained yet. Please call 'fit' with appropriate arguments before calling 'predict'.")
        return predictions
    
    def predict_proba(self, X):
        # 确保模型已经训练
        if hasattr(self, 'Model'):
            # 使用模型进行预测
            y_proba = self.Model.predict_proba(X.astype(self.dtype).copy())
            return y_proba
        else:
            raise ValueError("Model is not trained yet. Please call 'fit' with appropriate arguments before calling 'predict_proba'.")

    def score(self, X, y):
        # 确保模型已经训练
        if hasattr(self, 'Model'):
            # 使用模型进行预测
            y_pred = self.predict(X)
            # 计算并返回准确率
            return accuracy_score(y.astype('int64'), y_pred)
        else:
            raise ValueError("Model is not trained yet. Please call 'fit' with appropriate arguments before calling 'score'.")
    
    def transform(self, X, y=None):
        # 确保模型已经训练
        if hasattr(self, 'Model'):
            X = X.astype(self.dtype).copy()
            # 转换数据
            if self.Process is not None:
                X = self.Process.transform(X)
            
            # 转换数据为PyTorch张量
            X = torch.from_numpy(X).to(self.device)
            
            # 设置模型为评估模式
            self.Net.module.eval()
            with torch.no_grad():
                if self.nn_name in ['LMDANet']:
                    X = X.unsqueeze(1)  # 4D
                    X = torch.einsum('bdcw, hdc->bhcw', X, self.Net.module.channel_weight)  # 导联权重筛选
                    self.Net.module.eval()

                    X_time = self.Net.module.time_conv(X)  # batch, depth1, channel, samples_
                    X_time = self.Net.module.depthAttention(X_time)  # DA1
                    X = self.Net.module.chanel_conv(X_time)  # batch, depth2, 1, samples_
                    fc_input = self.Net.module.norm(X)
                
                elif self.nn_name in ['FBCNet', 'oFBCNet']:
                    X = torch.squeeze(X.permute((0,4,2,3,1)), dim = 4)
                    model_without_fc = torch.nn.Sequential(*list(self.Net.module.children())[:-1]).to(self.device)
                    fc_input = model_without_fc(X)
                
                elif self.nn_name in ['Tensor_CSPNet', 'oTensor_CSPNet']:
                    window_num, band_num = X.shape[1], X.shape[2]
                    X = X.reshape(X.shape[0], window_num*band_num, X.shape[3], X.shape[4])
                    X_csp = self.Net.module.BiMap_Block(X)
                    X_log = self.Net.module.LogEig(X_csp)
                    X_vec = X_log.view(X_log.shape[0], 1, window_num, -1)
                    fc_input = self.Net.module.Temporal_Block(X_vec).reshape(X.shape[0], -1)

                elif self.nn_name in ['EEGNetv4','ShallowFBCSPNet','Deep4Net','Graph_CSPNet', 'oGraph_CSPNet']:
                    model_without_fc = torch.nn.Sequential(*list(self.Net.module.children())[:-1]).to(self.device)
                    fc_input = model_without_fc(X)
                
                elif self.nn_name in ['EEGNet', 'ShallowNet']:
                    self.Net.module.model.eval()
                    model_without_fc = torch.nn.Sequential(*list(self.Net.module.model.children())[:-1]).to(self.device)
                    fc_input = model_without_fc(X.unsqueeze(1))
                
                elif self.nn_name in ['DeepNet']:
                    model_without_fc = torch.nn.Sequential(*list(self.Net.module.children())[:-3]).to(self.device)
                    fc_input = model_without_fc(X)
                
                else:
                    raise ValueError(f"{self.model_name} is not supported !")

            features = torch.flatten(fc_input, start_dim= 1).cpu().numpy()
            return features

        else:
            raise ValueError("Model is not trained yet. Please call 'fit' with appropriate arguments before calling 'transform'.")
