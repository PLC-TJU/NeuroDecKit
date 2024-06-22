import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# 假设数据如下
np.random.seed(42)  # 保持随机一致性

# 差被试
poor_none = np.random.rand(72) * 100
poor_rsf = np.random.rand(72) * 100

# 中等被试
medium_none = np.random.rand(150) * 100
medium_rsf = np.random.rand(150) * 100

# 良好被试
good_none = np.random.rand(43) * 100
good_rsf = np.random.rand(43) * 100

# 创建DataFrame
data = {
    'Accuracy': np.concatenate([poor_none, poor_rsf, medium_none, medium_rsf, good_none, good_rsf]),
    'Group': ['Poor']*72 + ['Poor']*72 + ['Medium']*150 + ['Medium']*150 + ['Good']*43 + ['Good']*43,
    'Method': ['None']*72 + ['RSF']*72 + ['None']*150 + ['RSF']*150 + ['None']*43 + ['RSF']*43
}
df = pd.DataFrame(data)

# 计算显著性差异的p值
p_values = {
    'Poor': ttest_ind(poor_none, poor_rsf).pvalue,
    'Medium': ttest_ind(medium_none, medium_rsf).pvalue,
    'Good': ttest_ind(good_none, good_rsf).pvalue
}

# 绘制半箱线半小提琴图
fig, axes = plt.subplots(1, 6, figsize=(18, 8), sharey=True)

# 定义绘图函数
def half_violinplot(ax, data, color, position, method):
    sns.violinplot(data=data, ax=ax, color=color, cut=0, inner=None)
    for violin in ax.collections:
        path = violin.get_paths()[0]
        path.vertices[:, 0] = np.clip(path.vertices[:, 0], -np.inf if method == 'None' else 0, np.inf)
        
def half_boxplot(ax, data, color, position, method):
    sns.boxplot(data=data, ax=ax, color=color, width=0.5, showcaps=False, showfliers=False, whiskerprops={'linewidth':0})
    for box in ax.artists:
        path = box.get_path()
        path.vertices[:, 0] = np.clip(path.vertices[:, 0], -np.inf if method == 'RSF' else 0, np.inf)

# 绘制每个组的图
for i, group in enumerate(['Poor', 'Medium', 'Good']):
    for j, method in enumerate(['None', 'RSF']):
        position = i * 2 + j
        ax = axes[position]
        data = df[(df['Group'] == group) & (df['Method'] == method)]['Accuracy'].values
        half_violinplot(ax, data, 'b' if method == 'None' else 'r', position, method)
        half_boxplot(ax, data, 'b' if method == 'None' else 'r', position, method)
        if j == 0:
            ax.set_ylabel('Accuracy')
        ax.set_title(f'{group} ({method})')
        ax.set_xticks([])

# 添加p值标注
for i, group in enumerate(['Poor', 'Medium', 'Good']):
    p_value = p_values[group]
    x1, x2 = i * 2, i * 2 + 1
    y, h, col = df['Accuracy'].max() + 5, 5, 'k'
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col, clip_on=False)
    plt.text((x1 + x2) * .5, y + h, f"p = {p_value:.3e}", ha='center', va='bottom', color=col)

plt.tight_layout()
plt.show()
