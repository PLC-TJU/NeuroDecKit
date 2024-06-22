# Authors: Pan.LC <panlincong@tju.edu.cn>
# Date: 2024/4/7
# License: MIT License

import numpy as np
from typing import List, Union, Optional
from numpy import ndarray

def _combine_datasets(Xt: ndarray, yt: ndarray, Xs: Optional[Union[ndarray, List[ndarray]]] = None, ys: 
    Optional[Union[ndarray, List[ndarray]]] = None) -> (List[ndarray], List[ndarray]): # type: ignore
    """
    结合来自不同来源的数据集。

    参数:
    Xt (ndarray): 目标域的数据集，形状为 (样本数, 通道数, 采样点数)。
    yt (ndarray): 目标域的标签向量，长度为样本数。
    Xs (Optional[Union[ndarray, List[ndarray]]], 默认为None): 源域的数据集，可以是以下之一:
        - None: 不使用源域数据。
        - 3维ndarray: 单个源域的数据集，形状为 (样本数, 通道数, 采样点数)。
        - 列表: 包含多个源域数据集的列表，每个元素都是一个3维ndarray。
        - 4维ndarray: 包含多个源域数据集的4维数组，形状为 (数据集数量, 样本数, 通道数, 采样点数)。
    ys (Optional[Union[ndarray, List[ndarray]]], 默认为None): 源域的标签集，可以是以下之一:
        - None: 不使用源域标签。
        - 1维ndarray: 单个源域的标签向量，长度为样本数。
        - 列表: 包含多个源域标签向量的列表，每个元素都是一个1维ndarray。

    返回:
    (List[ndarray], List[ndarray]): 两个列表，第一个是数据集列表，第二个是对应的标签列表。每个列表的元素数量等于“数据集来源数”。

    示例:
    >>> Xt = np.random.rand(10, 5, 100)  # 目标域数据集
    >>> yt = np.random.randint(0, 2, 10)  # 目标域标签
    >>> Xs = [np.random.rand(8, 5, 100), np.random.rand(12, 5, 100)]  # 源域数据集列表
    >>> ys = [np.random.randint(0, 2, 8), np.random.randint(0, 2, 12)]  # 源域标签列表
    >>> X_combined, Y_combined = combine_datasets(Xt, yt, Xs, ys)
    """

    X_combined = [Xt]  # 初始化数据集列表，首先添加目标域数据集
    Y_combined = [yt]  # 初始化标签列表，首先添加目标域标签

    # 检查Xs是否为None
    if Xs is None:
        return X_combined, Y_combined

    # 检查Xs是否为列表
    if isinstance(Xs, list):
        # 确保ys也是列表且长度与Xs相同
        if not isinstance(ys, list) or len(ys) != len(Xs):
            raise ValueError("ys must be a list with the same length as Xs")
        X_combined.extend(Xs)
        Y_combined.extend(ys)

    # 检查Xs是否为4维数组
    elif Xs.ndim == 4:
        # 将4维数组拆分为多个3维数组并添加到列表中
        for i in range(Xs.shape[0]):
            X_combined.append(Xs[i])
            Y_combined.append(ys[i])

    # 检查Xs是否为3维数组
    elif Xs.ndim == 3:
        X_combined.append(Xs)
        Y_combined.append(ys)

    # 其他情况，抛出异常
    else:
        raise ValueError("Xs must be either a 3D or 4D ndarray or a list of 3D ndarrays")

    return X_combined, Y_combined

# 与decode_domains搭配使用: from pyriemann.transfer import decode_domains
def combine_and_encode_datasets(Xt: ndarray, yt: ndarray, Xs: Optional[Union[ndarray, List[ndarray]]] = None, ys:
    Optional[Union[ndarray, List[ndarray]]] = None, Tags: Optional[List[str]] = None) -> (ndarray, ndarray): # type: ignore
    """
    组合来自不同源域的数据集，并对标签进行编码以区分不同的源域。

    参数:
    Xt (ndarray): 目标域的数据集，形状为 (样本数, 通道数, 采样点数)。
    yt (ndarray): 目标域的标签向量，长度为样本数。
    Xs (Optional[Union[ndarray, List[ndarray]]], 默认为None): 源域的数据集。
    ys (Optional[Union[ndarray, List[ndarray]]], 默认为None): 源域的标签集。
    Tags (Optional[List[str]], 默认为None): 定义不同来源数据集的标记列表。

    返回:
    (ndarray, ndarray): 所有数据集来源的数据集的组合的三维数组X，以及带有不同数据集domain标记的标签。
    """
    # 如果Tags为空，则生成默认的Tags列表
    if Tags is None:
        num_datasets = 1 if Xs is None else (len(Xs) + 1 if isinstance(Xs, list) else Xs.shape[0] + 1)
        Tags = ['S' + str(i+1) for i in range(num_datasets)]

    # 调用combine_datasets函数合并数据集
    X_combined, Y_combined = _combine_datasets(Xt, yt, Xs, ys)

    # 将合并后的数据集转换为三维数组
    X = np.concatenate(X_combined, axis=0)
    
    # 创建一个空列表来存储编码后的标签
    y_enc = []
    
    # 对每个源域的数据集进行编码
    for i, (X_domain, y_domain) in enumerate(zip(X_combined, Y_combined)):
        domain_tag = Tags[i]
        y_enc.extend([domain_tag + '/' + str(label) for label in y_domain])
    
    return X, np.array(y_enc)

# 与decode_domains搭配使用: from pyriemann.transfer import decode_domains
def encode_datasets(X: Union[List[ndarray], ndarray], 
                    Y: Union[List[ndarray], ndarray], 
                    domain_tags: Optional[List[str]] = None) -> (ndarray, ndarray, List[str]):  # type: ignore
    """
    将来自不同源域的数据集整理并编码标签。

    参数:
    X (Union[List[ndarray], ndarray]): 不同来源的数据集，可以是列表或4维数组。
    Y (Union[List[ndarray], ndarray]): 相应的不同来源的标签集，可以是列表或2维数组。
    domain_tags (Optional[List[str]], 默认为None): 各个来源数据集的标记，长度应与数据集数量一致。

    返回:
    (ndarray, ndarray): 整理后的数据集X和编码后的标签y_enc。
    """
    # 如果domain_tags为空，则生成默认的domain_tags列表
    if domain_tags is None:
        num_datasets = len(X) if isinstance(X, list) else X.shape[0]
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

#引用自 pyriemann.transfer import decode_domains
def decode_domains(X_enc, y_enc):
    """Decode the domains of the matrices in the labels.

    We handle the possibility of having different domains for the datasets by
    encoding the domain information into the labels of the matrices. This
    method converts the data into its original form, with a separate data
    structure for labels and for domains.

    Parameters
    ----------
    X_enc : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y_enc : ndarray, shape (n_matrices,)
        Extended labels for each matrix.

    Returns
    -------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    y : ndarray, shape (n_matrices,)
        Labels for each matrix.
    domain : ndarray, shape (n_matrices,)
        Domains for each matrix.

    See Also
    --------
    encode_domains

    Notes
    -----
    .. versionadded:: 0.4
    """
    y, domain = [], []
    for y_enc_ in y_enc:
        y_dec_ = y_enc_.split('/')
        domain.append(y_dec_[-2])
        y.append(y_dec_[-1])
        y = [int(i) for i in y]
    return X_enc, np.array(y), np.array(domain)