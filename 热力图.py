import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 使用您提供的实际数据
# 请将以下数据替换为您的实际数据
models = ['模型1', '模型2', '模型3', '模型4', '模型5', '模型6', '模型7', '模型8']
conditions = ['无提示词工程', '有提示词工程']
metrics = ['输出波动性', '一致性比率', '平均创造性得分', '一题多解能力评分']

# 实际数据应该是一个8x2x4的三维数组
# 这里使用您提供的部分数据作为示例
actual_data = np.array([
    # 无提示词工程数据
    [[0,0,3.8,0.170916598
],  # 模型1
     [0.097979798,0.75,3.8405,0.455092593
],  # 模型2
     [0,0,1.8,0
],  # 模型3
     [0.020833333,1,6.101388889,0.363741096
],  # 模型4
     [0,0.666666667,4.566777778,0.539426798
],  # 模型5
     [0.147660819,0.5,5.304666667,0.536933737
],  # 模型6
     [0,0,3.3,0.126286783
],  # 模型7
     [0.064327485,0.916666667,6.286722222,0.584982861
]],  # 模型8

    # 有提示词工程数据
    [[0.075757576,0.75,4.796777778,0.417609975
],  # 模型1
     [0.117424242,0.75,4.483555556,0.342614827
],  # 模型2
     [0,0,1.8,0
],  # 模型3
     [0.112794613,0.916666667,5.5605,0.474154038
],  # 模型4
     [0.0875,0.916666667,5.406277778,0.540320993
],  # 模型5
     [0,0.333333333,5.456277778,0.517994562
],  # 模型6
     [0,0,3.4,0.165918242
],  # 模型7
     [0.00877193,1,6.555888889,0.620080883
]]  # 模型8
])

# 转置数据以适应我们的需求 (8个模型 × 2种条件 × 4个指标)
actual_data = actual_data.transpose(1, 0, 2)

# 创建4个子图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('各模型在有无提示词工程下的性能热力图', fontsize=16, fontweight='bold')

# 定义绿色到红色的颜色映射
colors = ["green", "yellow", "red"]
cmap = LinearSegmentedColormap.from_list("green_to_red", colors)

# 为每个指标创建一个热力图
for i, metric in enumerate(metrics):
    # 确定子图位置
    row, col = i // 2, i % 2
    ax = axes[row, col]

    # 提取当前指标的数据
    metric_data = actual_data[:, :, i]

    # 创建DataFrame以便使用seaborn绘制热力图
    df_heatmap = pd.DataFrame(metric_data, index=models, columns=conditions)

    # 绘制热力图
    sns.heatmap(df_heatmap,
                ax=ax,
                cmap=cmap,
                annot=True,  # 在单元格中显示数值
                fmt=".3f",  # 数值格式，保留三位小数
                cbar_kws={'label': '数值'},  # 颜色条标签
                linewidths=0.5,  # 单元格之间的线宽
                annot_kws={"size": 10})  # 数值字体大小

    # 设置标题和标签
    ax.set_title(f'{metric}', fontsize=12, pad=10)

    # 特别设置输出波动性的Y轴标签
    if metric == '输出波动性':
        ax.set_ylabel('模型\n(输出波动性越低越稳定)', fontsize=10)
    else:
        ax.set_ylabel('模型', fontsize=10)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.93)  # 为总标题留出空间
plt.show()