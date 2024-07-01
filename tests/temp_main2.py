""" 
在这个改进后的实现中，函数 encode_datasets 现在可以处理以下几种情况：

单个三维数组 X 和相应的标签 Y。
列表形式的多个三维数组 X 和相应的标签列表 Y。
单个四维数组 X 和相应的标签列表 Y。
函数会根据输入数据的类型和形状来处理并编码标签，并且会生成默认的 domain_tags（如果没有提供的话）。
这样可以确保输入的数据集和标签在不同的情况下都能被正确处理和编码。
"""


import numpy as np
from typing import Union, List, Optional
from numpy import ndarray

def encode_datasets(X: Union[List[ndarray], ndarray], 
                    Y: Union[List[ndarray], ndarray], 
                    domain_tags: Optional[List[str]] = None) -> (ndarray, ndarray, List[str]): # type: ignore
    """
    将来自不同源域的数据集整理并编码标签。

    参数:
    X (Union[List[ndarray], ndarray]): 不同来源的数据集，可以是列表或3/4维数组。
    Y (Union[List[ndarray], ndarray]): 相应的不同来源的标签集，可以是列表或1/2维数组。
    domain_tags (Optional[List[str]], 默认为None): 各个来源数据集的标记，长度应与数据集数量一致。

    返回:
    (ndarray, ndarray, List[str]): 整理后的数据集X和编码后的标签y_enc及domain_tags。
    """
    
    # 如果domain_tags为空，则生成默认的domain_tags列表
    if domain_tags is None:
        num_datasets = len(X) if isinstance(X, list) or X.ndim == 4 else 1
        domain_tags = ['S' + str(i+1) for i in range(num_datasets)]

    # 初始化编码后的数据集和标签列表
    X_encoded = []
    y_encoded = []

    # 检查X是否为列表
    if isinstance(X, list):
        # 确保Y也是列表且长度与X相同
        if not isinstance(Y, list) or len(Y) != len(X):
            raise ValueError("Y must be a list with the same length as X")
        for i, (X_i, Y_i) in enumerate(zip(X, Y)):
            X_encoded.append(X_i)
            y_encoded.extend([domain_tags[i] + '/' + str(y) for y in Y_i])
    
    # 检查X是否为4维数组
    elif X.ndim == 4:
        for i in range(X.shape[0]):
            X_encoded.append(X[i])
            y_encoded.extend([domain_tags[i] + '/' + str(y) for y in Y[i]])
    
    # 检查X是否为3维数组
    elif X.ndim == 3:
        X_encoded.append(X)
        y_encoded.extend([domain_tags[0] + '/' + str(y) for y in Y])

    # 其他情况，抛出异常
    else:
        raise ValueError("X must be either a list of 3D ndarrays, a 4D ndarray, or a single 3D ndarray")

    # 将编码后的数据集转换为三维数组
    X_encoded = np.concatenate(X_encoded, axis=0)
    
    return X_encoded, np.array(y_encoded), domain_tags

# 示例使用
X_3d = np.random.randn(10, 5, 100)  # 单个3维数组
Y_3d = np.random.randint(0, 2, 10)  # 对应的标签

X_list = [np.random.randn(10, 5, 100), np.random.randn(15, 5, 100)]  # 列表形式的多个3维数组
Y_list = [np.random.randint(0, 2, 10), np.random.randint(0, 2, 15)]  # 对应的标签列表

X_4d = np.random.randn(2, 10, 5, 100)  # 单个4维数组
Y_4d = [np.random.randint(0, 2, 10), np.random.randint(0, 2, 10)]  # 对应的标签列表

# 调用函数
X_encoded, y_encoded, domain_tags = encode_datasets(X_3d, Y_3d)
print("Encoded X (3D):", X_encoded.shape)
print("Encoded y (3D):", y_encoded)

X_encoded, y_encoded, domain_tags = encode_datasets(X_list, Y_list)
print("Encoded X (List):", X_encoded.shape)
print("Encoded y (List):", y_encoded)

X_encoded, y_encoded, domain_tags = encode_datasets(X_4d, Y_4d)
print("Encoded X (4D):", X_encoded.shape)
print("Encoded y (4D):", y_encoded)

a=0
