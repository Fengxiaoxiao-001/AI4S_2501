"""
综合得分 = (指标1得分 × 0.25) + (指标2得分 × 0.25) + (指标3得分 × 0.25) + (指标4得分 × 0.25)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据（只使用前4列，对应四个指标）
data = {
    "简单难度-无提示词工程": [0.625, 0.022724781, 3.868479167, 0.433063709],  # 一致性比率, 输出波动性, 平均创造性得分, 一题多解能力评分
    "中等难度-无提示词工程": [0.53125, 0.052083333, 4.940958333, 0.279662414],
    "困难难度-无提示词工程": [0.28125, 0.049242424, 4.315583333, 0.328791551],
    "简单难度-有提示词工程": [0.75, 0.011101974, 4.3288125, 0.345749176],
    "中等难度-有提示词工程": [0.625, 0.013888889, 5.430708333, 0.341995108],
    "困难难度-有提示词工程": [0.375, 0.125852273, 4.287708333, 0.466765786]
}

# 定义指标名称和评分方向
indicators = {
    "一致性比率": 1,  # 正向指标，值越大越好
    "输出波动性": -1,  # 负向指标，值越小越好
    "平均创造性得分": 1,  # 正向指标，值越大越好
    "一题多解能力评分": 1  # 正向指标，值越大越好
}

# 创建DataFrame以便处理
df_raw = pd.DataFrame(data).T
df_raw.columns = ["一致性比率", "输出波动性", "平均创造性得分", "一题多解能力评分"]

print("原始数据:")
print(df_raw)
print()


# 对每个指标单独进行归一化到1-10分
def normalize_to_1_10(series, indicator_type):
    """将单个指标的数据归一化到1-10分"""
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:  # 防止除以零
        return pd.Series([5.5] * len(series), index=series.index)

    if indicator_type == 1:  # 正向指标
        normalized = (series - min_val) / (max_val - min_val)
    else:  # 负向指标
        normalized = (max_val - series) / (max_val - min_val)

    # 映射到1-10分
    scores = 1 + 9 * normalized
    return scores.round(2)


# 对每个指标应用归一化
df_normalized = pd.DataFrame()
for col in df_raw.columns:
    df_normalized[col] = normalize_to_1_10(df_raw[col], indicators[col])

print("归一化后的得分(1-10分):")
print(df_normalized)
print()

# 计算综合得分（等权重平均）
df_normalized["综合得分"] = df_normalized.mean(axis=1).round(2)

print("综合得分:")
print(df_normalized["综合得分"])
print()

# 可视化结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('提示词工程对不同难度问题解决效果的影响分析', fontsize=16, fontweight='bold')

# 1. 雷达图 - 显示四个维度的评分结果
categories = list(indicators.keys())
N = len(categories)

# 计算每个维度的角度
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 设置雷达图
ax1 = plt.subplot(121, polar=True)
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories)
ax1.set_ylim(0, 10)
ax1.set_yticks([2, 4, 6, 8, 10])
ax1.set_yticklabels(['2', '4', '6', '8', '10'])
ax1.grid(True, alpha=0.3)

# 定义颜色 - 不同难度使用不同颜色
colors = {
    "简单难度-无提示词工程": "lightblue",
    "简单难度-有提示词工程": "blue",
    "中等难度-无提示词工程": "orange",
    "中等难度-有提示词工程": "red",
    "困难难度-无提示词工程": "lightgreen",
    "困难难度-有提示词工程": "green"
}

colors_radar = {
    "简单难度-无提示词工程": "blue",
    "简单难度-有提示词工程": "blue",
    "中等难度-无提示词工程": "red",
    "中等难度-有提示词工程": "red",
    "困难难度-无提示词工程": "green",
    "困难难度-有提示词工程": "green"
}

# 准备雷达图数据
radar_data = {}
for condition in df_normalized.index:
    values = df_normalized.loc[condition, categories].tolist()
    values += values[:1]  # 闭合图形
    radar_data[condition] = values

# 绘制每个条件的雷达图 - 修改部分：添加线型区分
for condition, values in radar_data.items():
    # 根据有无提示词工程设置线型：有提示词用实线，无提示词用虚线
    if "有提示词" in condition:
        linestyle = '-'  # 实线
    else:
        linestyle = '--'  # 虚线

    # 绘制雷达图线条和填充
    ax1.plot(angles, values, marker='', linestyle=linestyle, linewidth=2,
             label=condition, color=colors_radar[condition])
    ax1.fill(angles, values, alpha=0.1, color=colors[condition])

ax1.set_title('不同难度下提示词工程效果雷达图(1-10分)', size=12, pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 2. 柱状图 - 显示综合得分对比
conditions = list(df_normalized.index)
scores = df_normalized["综合得分"].values

# 设置颜色
bar_colors = [colors[cond] for cond in conditions]

bars = ax2.bar(conditions, scores, color=bar_colors, alpha=0.7)
ax2.set_ylabel('综合得分(1-10分)')
ax2.set_title('不同条件下综合得分对比')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, axis='y', alpha=0.3)

# 在柱子上方添加数值标签
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
             f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 设置y轴范围，留出空间显示标签
ax2.set_ylim(0, max(scores) * 1.1)

plt.tight_layout()
plt.show()

# 输出分析结论
print("=" * 60)
print("分析结论")
print("=" * 60)

# 比较有提示词工程和无提示词工程的效果
for difficulty in ["简单难度", "中等难度", "困难难度"]:
    with_prompt = df_normalized.loc[f"{difficulty}-有提示词工程", "综合得分"]
    without_prompt = df_normalized.loc[f"{difficulty}-无提示词工程", "综合得分"]

    improvement = with_prompt - without_prompt
    improvement_pct = (improvement / without_prompt) * 100

    print(f"\n{difficulty}:")
    print(f"  无提示词工程综合得分: {without_prompt:.2f}")
    print(f"  有提示词工程综合得分: {with_prompt:.2f}")

    if improvement > 0:
        print(f"  提示词工程提升: +{improvement:.2f} 分 (+{improvement_pct:.1f}%)")
    else:
        print(f"  提示词工程效果: {improvement:.2f} 分 ({improvement_pct:.1f}%)")

# 找出最佳和最差表现
best_condition = df_normalized["综合得分"].idxmax()
best_score = df_normalized["综合得分"].max()
worst_condition = df_normalized["综合得分"].idxmin()
worst_score = df_normalized["综合得分"].min()

print(f"\n最佳表现: {best_condition} (得分: {best_score:.2f})")
print(f"最差表现: {worst_condition} (得分: {worst_score:.2f})")

# 检查每个指标的最佳表现
print("\n各指标最佳表现:")
for indicator in categories:
    best_for_indicator = df_normalized[indicator].idxmax()
    best_score = df_normalized[indicator].max()
    print(f"  {indicator}: {best_for_indicator} ({best_score:.2f}分)")