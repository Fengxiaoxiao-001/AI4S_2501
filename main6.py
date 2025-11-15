import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 无提示词工程数据
no_prompt_data = {
    '简单': {
        'Model1': 1,
        'Model2': 0.95,
        'Model3': 1,
        'Model4': 0.95,
        'Model5': 1,
        'Model6': 0.975,
        'Model7': 1,
        'Model8': 0.975
    },
    '中等': {
        'Model1': 1,
        'Model2': 1,
        'Model3': 1,
        'Model4': 1,
        'Model5': 1,
        'Model6': 0.8,
        'Model7': 1,
        'Model8': 0.875
    },
    '困难': {
        'Model1': 1,
        'Model2': 0.875,
        'Model3': 1,
        'Model4': 1,
        'Model5': 1,
        'Model6': 0.925,
        'Model7': 1,
        'Model8': 1
    }
}

# 有提示词工程数据
with_prompt_data = {
    '简单': {
        'Model1': 1,
        'Model2': 1,
        'Model3': 1,
        'Model4': 1,
        'Model5': 0.95,
        'Model6': 1,
        'Model7': 1,
        'Model8': 0.975
    },
    '中等': {
        'Model1': 1,
        'Model2': 1,
        'Model3': 1,
        'Model4': 0.9,
        'Model5': 1,
        'Model6': 1,
        'Model7': 1,
        'Model8': 1
    },
    '困难': {
        'Model1': 0.875,
        'Model2': 0.825,
        'Model3': 1,
        'Model4': 0.875,
        'Model5': 0.9,
        'Model6': 1,
        'Model7': 1,
        'Model8': 1
    }
}

# 计算每个模型在不同难度下的平均stability_score
models = ['Model1', 'Model2', 'Model3', 'Model4', 'Model5', 'Model6', 'Model7', 'Model8']

# 计算无提示词工程的平均值
no_prompt_avg = []
for model in models:
    scores = [no_prompt_data[difficulty][model] for difficulty in ['简单', '中等', '困难']]
    no_prompt_avg.append(np.mean(scores))

# 计算有提示词工程的平均值
with_prompt_avg = []
for model in models:
    scores = [with_prompt_data[difficulty][model] for difficulty in ['简单', '中等', '困难']]
    with_prompt_avg.append(np.mean(scores))

# 计算提升幅度（有提示词 - 无提示词）
improvement = [with_prompt_avg[i] - no_prompt_avg[i] for i in range(len(models))]

# 创建图表和主y轴
fig, ax1 = plt.subplots(figsize=(12, 7))  # 创建画布和主轴ax1

# 在主y轴（ax1）上绘制条形图
x = np.arange(len(models))
width = 0.35
bars1 = ax1.bar(x - width/2, no_prompt_avg, width, label='无提示词工程', color='skyblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, with_prompt_avg, width, label='有提示词工程', color='lightcoral', alpha=0.8)

# 设置主y轴（左侧）标签和范围
ax1.set_ylabel('稳定性得分', fontsize=12)
ax1.set_ylim(0.7, 1.1)  # 固定条形图范围
ax1.set_xlabel('模型', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.grid(True, alpha=0.3, axis='y')  # 网格仅应用于主y轴
ax1.set_title('各模型在有无提示词工程下的稳定性得分对比', fontsize=16, fontweight='bold', pad=20)

# 创建次y轴（右侧）用于折线图
ax2 = ax1.twinx()  # 关键步骤：创建双y轴[1](@ref)
line = ax2.plot(x, improvement, marker='o', color='green', linewidth=2, label='提升幅度', markersize=8)
ax2.set_ylabel('提升幅度', fontsize=12, color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(-0.1, 0.1)  # 根据提升幅度数据设置范围，确保折线可见

# 合并图例：将两个轴的图例合并显示
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

# 添加数值标签（略，与原代码相同，但需将plt.text改为ax1.text和ax2.annotate）
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.005,
             f'{no_prompt_avg[i]:.3f}', ha='center', va='bottom', fontsize=8)
    ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.005,
             f'{with_prompt_avg[i]:.3f}', ha='center', va='bottom', fontsize=8)
for i, point in enumerate(improvement):
    ax2.annotate(f'{point:.3f}', (x[i], point), textcoords="offset points",
                 xytext=(0, 10 if point >= 0 else -20), ha='center', fontsize=9, color='green')

plt.tight_layout()
plt.show()
