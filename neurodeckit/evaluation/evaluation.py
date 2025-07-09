"""
Evaluation module for brain signal classification.

Author: LC.Pan <panlincong@tju.edu.cn.com>
Date: 2025/3/25
License: All rights reserved

"""

import numpy as np
import pandas as pd
import time
from abc import ABC, abstractmethod
from typing import Union, Dict, List, Optional, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, cohen_kappa_score, roc_curve
)
from ..transfer_learning import TLSplitter

class ClassificationMetrics:
    """统一管理和计算分类评估指标"""
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_prob: Optional[np.ndarray] = None,
                 n_points: int = 1000):
        """
        初始化分类评估指标
        
        参数:
        y_true : ndarray, 真实标签
        y_pred : ndarray, 预测标签
        y_prob : ndarray, 预测概率（可选）
        n_points : int, ROC曲线的点数，默认为1000
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.classes = np.unique(y_true)
        self.n_classes = len(self.classes)
        self.n_points = n_points  # 控制ROC曲线的点数
        
        # 所有支持的指标列表
        self.all_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 
            'kappa', 'auc', 'roc_curve'
        ]
    
    def calculate(self, metrics: Union[str, List[str]] = None) -> Dict[str, Any]:
        """
        计算指定的评估指标
        
        参数:
        metrics : str | List[str], 需要计算的指标列表或单个指标名称
                  可以是'all'表示所有指标，或单个字符串如'auc'
        
        返回:
        results : 包含计算结果的字典
        """
        # 处理输入参数
        if metrics is None:
            metrics = ['accuracy']
        elif metrics == 'all':
            metrics = self.all_metrics.copy()
        elif isinstance(metrics, str):
            metrics = [metrics]
        
        # 检查无效指标
        valid_metrics = set(metrics) & set(self.all_metrics)
        if not valid_metrics:
            raise ValueError(f"没有有效的指标。支持的指标: {self.all_metrics}")
        
        results = {}
        
        # 基础指标
        if 'accuracy' in metrics or 'acc' in metrics:
            results['accuracy'] = np.mean(self.y_pred == self.y_true)
        
        # 分类指标
        if any(m in metrics for m in ['precision', 'recall', 'f1']):
            # 多分类使用宏平均
            average = 'macro' if self.n_classes > 2 else 'binary'
            
            if 'precision' in metrics:
                results['precision'] = precision_score(self.y_true, self.y_pred, average=average)
            
            if 'recall' in metrics:
                results['recall'] = recall_score(self.y_true, self.y_pred, average=average)
            
            if 'f1' in metrics:
                results['f1'] = f1_score(self.y_true, self.y_pred, average=average)
        
        # Kappa系数
        if 'kappa' in metrics:
            results['kappa'] = cohen_kappa_score(self.y_true, self.y_pred)
        
        # ROC相关指标
        roc_metrics_requested = any(m in metrics for m in ['auc', 'roc_curve'])
        if roc_metrics_requested and self.y_prob is not None:
            # 二分类ROC
            if self.n_classes == 2:
                if 'auc' in metrics:
                    results['auc'] = roc_auc_score(self.y_true, self.y_prob[:, 1])
                
                if 'roc_curve' in metrics:
                    # 首先获取原始ROC曲线点
                    fpr, tpr, thresholds = roc_curve(
                        self.y_true, self.y_prob[:, 1], 
                        drop_intermediate=False
                    )
                    
                    # 创建均匀分布的fpr点（从0到1）
                    new_fpr = np.linspace(0, 1, self.n_points)
                    
                    # 使用线性插值计算对应的tpr
                    new_tpr = np.interp(new_fpr, fpr, tpr)
                    
                    # 近似插值阈值
                    new_thresholds = np.interp(new_fpr, fpr, thresholds)
                    
                    results['roc_curve'] = {
                        'fpr': new_fpr,
                        'tpr': new_tpr,
                        'thresholds': new_thresholds
                    }
            
            # 多分类ROC（OvR策略）
            elif self.n_classes > 2:
                if 'auc' in metrics:
                    # 多分类AUC使用OvR策略
                    results['auc'] = roc_auc_score(
                        self.y_true, self.y_prob, multi_class='ovr'
                    )
                
                if 'roc_curve' in metrics:
                    # 多分类ROC曲线计算每个类别的曲线
                    roc_data = {}
                    for i, cls in enumerate(self.classes):
                        y_true_binary = (self.y_true == cls).astype(int)
                        
                        # 首先获取原始ROC曲线点
                        fpr, tpr, thresholds = roc_curve(
                            y_true_binary, self.y_prob[:, i], 
                            drop_intermediate=False
                        )
                        
                        # 创建均匀分布的fpr点（从0到1）
                        new_fpr = np.linspace(0, 1, self.n_points)
                        
                        # 使用线性插值计算对应的tpr
                        new_tpr = np.interp(new_fpr, fpr, tpr)
                        
                        # 近似插值阈值
                        new_thresholds = np.interp(new_fpr, fpr, thresholds)
                        
                        roc_data[f"class_{cls}"] = {
                            'fpr': new_fpr,
                            'tpr': new_tpr,
                            'thresholds': new_thresholds
                        }
                    results['roc_curve'] = roc_data
        
        # 处理ROC指标请求但y_prob不可用的情况
        elif roc_metrics_requested and self.y_prob is None:
            if 'auc' in metrics:
                results['auc'] = np.nan
                print("警告: 无法计算AUC，模型不支持概率预测")
            if 'roc_curve' in metrics:
                results['roc_curve'] = None
                print("警告: 无法计算ROC曲线，模型不支持概率预测")
        
        # 处理不支持的指标
        for metric in metrics:
            if metric not in results:
                raise ValueError(f"不支持的指标: {metric}")
        
        # 返回请求的指标
        return {k: v for k, v in results.items() if k in metrics}

class BaseEvaluator(ABC):
    """
    脑电信号分类评估的基类
    包含公共参数设置和功能函数
    """
    def __init__(self, 
                 info: pd.DataFrame = None, 
                 cv: Union[float, Any] = None, 
                 metrics: Optional[List[str]] = ['accuracy'],
                ):
        """
        初始化评估器
        
        参数:
        info : DataFrame, (n_samples, 3)
            样本信息 [subject, session, block/run]
        cv : float | 交叉验证器对象, default=None
            目标域数据划分方式
        metrics : str | List[str], optional
            需要计算的指标列表或单个指标名称
            'accuracy', 'precision', 'recall', 'f1', 'kappa', 'auc', 'roc_curve'
            还可以是'all'表示所有指标，
            默认只计算'accuracy'
        """
        self.cv = cv
        
        if metrics == 'all':
            self.metrics = ['accuracy', 'precision', 'recall', 
                            'f1', 'kappa', 'auc', 'roc_curve']
        elif isinstance(metrics, str):
            self.metrics = [metrics]
        else:
            self.metrics = metrics

        if info is not None:
            # 验证输入数据
            self._validate_data(info)
            
            # 生成域标签
            domain_tags = self._generate_domain_tags(info)
            self.domain_tags = domain_tags  # 存储域标签
        else:
            self.domain_tags = None
    
    @abstractmethod
    def _validate_data(self, info: pd.DataFrame) -> None:
        """
        验证输入数据是否符合分类场景要求（由子类实现）
        
        参数:
        info : DataFrame
            样本信息，包含subject, session, block
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_domain_tags(self) -> List[str]:
        """
        获取所有独特的域标签
        
        返回:
        domain_tags : 所有独特的域标签列表
        """
        if self.domain_tags is None:
            raise RuntimeError("请先调用evaluate方法生成域标签")
        
        # 排序并去重
        domain_tags_unique = sorted(list(set(self.domain_tags)))
        return domain_tags_unique
    
    @abstractmethod
    def _generate_domain_tags(self, info: pd.DataFrame) -> List[str]:
        """
        根据场景生成域标签（由子类实现）
        
        参数:
        info : DataFrame
            样本信息，包含subject, session, block
        
        返回:
        domain_tags : 每个样本的域标签列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    @abstractmethod
    def _validate_target_domain(self, target_domain: str, domain_tags: List[str]) -> None:
        """
        验证目标域是否有效（由子类实现）
        
        参数:
        target_domain : str
            用户指定的目标域
        domain_tags : List[str]
            所有样本的域标签列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def evaluate(self, 
                data: np.ndarray, 
                label: np.ndarray, 
                pipelines: Union[Pipeline, Dict[str, Pipeline]],
                target_domain: str,
                exclude_domains: Optional[List[str]] = None,
                ) -> Dict:
        """
        评估分类性能
        
        参数:
        data : ndarray, (n_samples, n_channels, n_times)
            脑电信号数据
        label : ndarray, (n_samples,)
            样本标签
        pipelines : Pipeline | Dict[str, Pipeline]
            分类模型或模型字典
        target_domain : str
            目标域名称
        exclude_domains : List[str], optional
            需要排除的域列表
        
        返回:
        results : 分类结果字典，包含详细性能指标
        """
        
        # 检查域标签是否可用
        if self.domain_tags is None:
            raise RuntimeError("未初始化域标签，请确保在初始化时提供了info参数")

        # 验证目标域
        if target_domain is None or (exclude_domains is not None and target_domain in exclude_domains):
            raise ValueError("目标域不能为None且在exclude_domains中")
        
        domain_tags = self.domain_tags.copy()
        self._validate_target_domain(target_domain, domain_tags)
        
        # 排除域样本和标签
        if exclude_domains:
            # 创建排除掩码
            exclude_mask = np.isin(domain_tags, exclude_domains, invert=True)
            # 过滤数据和标签
            data = data[exclude_mask]
            label = label[exclude_mask]
            domain_tags = [d for i, d in enumerate(domain_tags) if exclude_mask[i]]
        
        # 验证目标域是否存在
        if target_domain not in domain_tags:
            raise ValueError(f"目标域 '{target_domain}' 在数据中不存在")
        
        # 创建扩展标签
        y_extended = np.array([f"{d}/{l}" for d, l in zip(domain_tags, label)])
        
        # 创建TLSplitter
        splitter = TLSplitter(
            target_domain=target_domain,
            cv=self.cv,
        )
        
        # 处理单个/多个模型
        if not isinstance(pipelines, dict):
            pipelines = {"single_pipeline": pipelines}
        
        results = {}
        for model_name, model in pipelines.items():
            # 对于迁移学习分类，确保模型是TL_Classifier对象 
            # if not isinstance(model, TL_Classifier):
            #     raise ValueError(f"模型 '{model_name}' 必须是 TL_Classifier 对象")
            
            # 初始化性能指标
            fold_results = {
                'train_time': [],
                'test_time': [],
                'train_samples': [],
                'test_samples': []
            }
            
            # 添加其他指标容器
            metrics = self.metrics
            if metrics is not None:
                for metric in metrics:
                    fold_results[metric] = []
            
            # 进行交叉验证
            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(data, y_extended)):
                X_train, y_train = data[train_idx], y_extended[train_idx]
                X_test, y_test = data[test_idx], y_extended[test_idx]
                
                # 提取真实标签
                y_train_true = np.array([int(y.split('/')[-1]) for y in y_train])
                y_test_true = np.array([int(y.split('/')[-1]) for y in y_test])
                
                # 训练模型并记录时间
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # 测试模型并记录时间
                start_time = time.time()
                y_pred = model.predict(X_test)
                test_time = time.time() - start_time
                
                # 尝试获取预测概率（用于AUC和ROC）
                y_prob = None
                try:
                    y_prob = model.predict_proba(X_test)
                except (AttributeError, NotImplementedError):
                    # 模型不支持概率预测
                    pass
                
                # 计算性能指标
                metrics_calculator = ClassificationMetrics(y_test_true, y_pred, y_prob)
                calculated_metrics = metrics_calculator.calculate(metrics)
                for metric in metrics:
                    if metric in calculated_metrics:
                        fold_results[metric].append(calculated_metrics[metric])
                
                # 存储时间和样本信息
                fold_results['train_time'].append(train_time)
                fold_results['test_time'].append(test_time)
                fold_results['train_samples'].append(len(X_train))
                fold_results['test_samples'].append(len(X_test))
            
            # 计算平均指标
            model_results = {}
            for metric, values in fold_results.items():
                # ROC曲线是字典类型，不计算平均值
                if metric == 'roc_curve':
                    model_results[metric] = values
                    continue
                
                # 计算数值指标的平均值和标准差
                if all(isinstance(v, (int, float)) or np.isnan(v) for v in values):
                    model_results[metric] = {
                        'values': values,
                        'mean': np.nanmean(values),
                        'std': np.nanstd(values)
                    }
                # 其他类型直接存储
                else:
                    model_results[metric] = values
            
            model_results['folds'] = len(fold_results['accuracy'])
            results[model_name] = model_results
            
        return results


class WithinSessionEvaluator(BaseEvaluator):
    """
    within-session分类评估
    同一session内划分训练集和测试集（非迁移学习）
    
    要求:
    - 所有样本来自同一个被试
    - 存在多个不同的session
    """
    def _validate_data(self, info: pd.DataFrame) -> None:
        """验证数据是否符合within-session分类要求"""
        subjects = info.iloc[:, 0].unique()
        
        if len(subjects) > 1:
            raise ValueError(f"WithinSessionEvaluator 要求所有样本来自同一个被试, 但找到 {len(subjects)} 个被试")
    
    def _generate_domain_tags(self, info: pd.DataFrame) -> List[str]:
        """生成session级别的域标签"""
        subjects = info.iloc[:, 0]
        sessions = info.iloc[:, 1]
        return [f"S{sub}_Sess{sess}" for sub, sess in zip(subjects, sessions)]
    
    def _validate_target_domain(self, target_domain: str, domain_tags: List[str]) -> None:
        """验证目标域是否有效"""
        if target_domain not in set(domain_tags):
            raise ValueError(f"目标域 '{target_domain}' 不存在于数据中")
        
        # 检查目标域样本数量
        target_count = sum(1 for tag in domain_tags if tag == target_domain)
        if target_count < 10:
            raise ValueError(f"目标域 '{target_domain}' 样本数量不足 ({target_count} 个样本)")


class WithinSubjectEvaluator(BaseEvaluator):
    """
    within-subject分类评估
    同一subject内划分训练集和测试集（非迁移学习）
    
    要求:
    - 无特殊要求
    """
    def _validate_data(self, info: pd.DataFrame) -> None:
        """无特殊数据要求"""
        pass
    
    def _generate_domain_tags(self, info: pd.DataFrame) -> List[str]:
        """生成subject级别的域标签"""
        subjects = info.iloc[:, 0]
        return [f"S{sub}" for sub in subjects]
    
    def _validate_target_domain(self, target_domain: str, domain_tags: List[str]) -> None:
        """验证目标域是否有效"""
        if target_domain not in set(domain_tags):
            raise ValueError(f"目标域 '{target_domain}' 不存在于数据中")
        
        # 检查目标域样本数量
        target_count = sum(1 for tag in domain_tags if tag == target_domain)
        if target_count < 10:
            raise ValueError(f"目标域 '{target_domain}' 样本数量不足 ({target_count} 个样本)")


class CrossSessionEvaluator(BaseEvaluator):
    """
    cross-session分类评估
    同一subject内不同session之间的迁移学习
    
    要求:
    - 所有样本来自同一个被试
    - 至少有两个不同的session
    """
    def _validate_data(self, info: pd.DataFrame) -> None:
        """验证数据是否符合cross-session分类要求"""
        subjects = info.iloc[:, 0].unique()
        sessions = info.iloc[:, 1].unique()
        
        if len(subjects) > 1:
            raise ValueError(f"CrossSessionEvaluator 要求所有样本来自同一个被试, 但找到 {len(subjects)} 个被试")
        
        if len(sessions) < 2:
            raise ValueError(f"CrossSessionEvaluator 要求至少有两个不同的session, 但只找到 {len(sessions)} 个session")
    
    def _generate_domain_tags(self, info: pd.DataFrame) -> List[str]:
        """生成session级别的域标签"""
        subjects = info.iloc[:, 0]
        sessions = info.iloc[:, 1]
        return [f"S{sub}_Sess{sess}" for sub, sess in zip(subjects, sessions)]
    
    def _validate_target_domain(self, target_domain: str, domain_tags: List[str]) -> None:
        """验证目标域是否有效"""
        if target_domain not in set(domain_tags):
            raise ValueError(f"目标域 '{target_domain}' 不存在于数据中")
        
        # 检查源域数量
        source_domains = set(tag for tag in domain_tags if tag != target_domain)
        if len(source_domains) == 0:
            raise ValueError(f"目标域 '{target_domain}' 没有可用的源域")
        
        # 检查目标域样本数量
        target_count = sum(1 for tag in domain_tags if tag == target_domain)
        if target_count < 10:
            raise ValueError(f"目标域 '{target_domain}' 样本数量不足 ({target_count} 个样本)")


class CrossSubjectEvaluator(BaseEvaluator):
    """
    cross-subject分类评估
    不同subject之间的迁移学习
    
    要求:
    - 至少有两个不同的subject
    """
    def _validate_data(self, info: pd.DataFrame) -> None:
        """验证数据是否符合cross-subject分类要求"""
        subjects = info.iloc[:, 0].unique()
        if len(subjects) < 2:
            raise ValueError(f"CrossSubjectEvaluator 要求至少有两个不同的被试, 但只找到 {len(subjects)} 个被试")
    
    def _generate_domain_tags(self, info: pd.DataFrame) -> List[str]:
        """生成subject级别的域标签"""
        subjects = info.iloc[:, 0]
        return [f"S{sub}" for sub in subjects]
    
    def _validate_target_domain(self, target_domain: str, domain_tags: List[str]) -> None:
        """验证目标域是否有效"""
        if target_domain not in set(domain_tags):
            raise ValueError(f"目标域 '{target_domain}' 不存在于数据中")
        
        # 检查源域数量
        source_domains = set(tag for tag in domain_tags if tag != target_domain)
        if len(source_domains) == 0:
            raise ValueError(f"目标域 '{target_domain}' 没有可用的源域")
        
        # 检查目标域样本数量
        target_count = sum(1 for tag in domain_tags if tag == target_domain)
        if target_count < 10:
            raise ValueError(f"目标域 '{target_domain}' 样本数量不足 ({target_count} 个样本)")


class CrossBlockEvaluator(BaseEvaluator):
    """
    cross-block分类评估
    同一session内不同block之间的迁移学习
    
    要求:
    - 所有样本来自同一个被试
    - 存在多个不同的block
    - 可以存在于多个session中
    """
    def _validate_data(self, info: pd.DataFrame) -> None:
        """验证数据是否符合cross-block分类要求"""
        subjects = info.iloc[:, 0].unique()
        sessions = info.iloc[:, 1].unique()
        blocks = info.iloc[:, 2].unique()
        
        if len(subjects) > 1:
            raise ValueError(f"CrossBlockEvaluator 要求所有样本来自同一个被试, 但找到 {len(subjects)} 个被试")
        
        if len(blocks) < 2 and len(sessions) < 2:
            raise ValueError(f"CrossBlockEvaluator 要求至少有两个不同的block, 但只找到 {len(blocks)} 个block")
    
    def _generate_domain_tags(self, info: pd.DataFrame) -> List[str]:
        """生成block级别的域标签"""
        subjects = info.iloc[:, 0]
        sessions = info.iloc[:, 1]
        blocks = info.iloc[:, 2]  # 第三列是block/run信息
        return [f"S{sub}_Sess{sess}_Block{block}" for sub, sess, block in zip(subjects, sessions, blocks)]
    
    def _validate_target_domain(self, target_domain: str, domain_tags: List[str]) -> None:
        """验证目标域是否有效"""
        if target_domain not in set(domain_tags):
            raise ValueError(f"目标域 '{target_domain}' 不存在于数据中")
        
        # 检查源域数量
        source_domains = set(tag for tag in domain_tags if tag != target_domain)
        if len(source_domains) == 0:
            raise ValueError(f"目标域 '{target_domain}' 没有可用的源域")
        
        # 检查目标域样本数量
        target_count = sum(1 for tag in domain_tags if tag == target_domain)
        if target_count < 5:
            raise ValueError(f"目标域 '{target_domain}' 样本数量不足 ({target_count} 个样本)")