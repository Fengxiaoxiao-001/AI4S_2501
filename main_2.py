import matplotlib.pyplot as plt

# 设置中文字体（确保系统有中文字体）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表和子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('提示词工程对大模型解题能力的影响分析', fontsize=16, fontweight='bold')

# 数据准备（根据您图片中的数值）
metrics_data = {
    '输出波动性': {
        '无提示词工程': 0.041350179,
        '有提示词工程': 0.050281045
    },
    '一致性比率': {
        '无提示词工程': 0.479166667,
        '有提示词工程': 0.583333333
    },
    '平均创造性得分': {
        '无提示词工程': 4.375006944,
        '有提示词工程': 4.682409722
    },
    '一题多解能力评分': {
        '无提示词工程': 0.347172558,
        '有提示词工程': 0.38483669
    }
}

# 子图标题
subplot_titles = [
    '输出波动性(数值波动性)（越低越稳定）',
    '一致性比率(质量稳定性)（越高越稳定）',
    '平均创造性得分（越高越有创造性）',
    '一题多解能力评分（越高能力越强）'
]

# 颜色设置
colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色
conditions = ['无提示词工程', '有提示词工程']

# 绘制每个子图
for i, (metric, title) in enumerate(zip(metrics_data.keys(), subplot_titles)):
    ax = axes[i // 2, i % 2]

    # 获取数据
    data = metrics_data[metric]
    values = [data['无提示词工程'], data['有提示词工程']]

    # 创建柱状图
    bars = ax.bar(conditions, values, color=colors, alpha=0.8)

    # 设置标题和标签
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel('得分', fontsize=10)

    # 设置y轴范围，留出空间显示数值标签
    y_max = max(values) * 1.2  # 只比最大值高20%，而不是固定值
    ax.set_ylim(0, y_max)

    # 在每个柱子上方添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        # 使用相对于柱高的偏移，而不是固定值
        offset = y_max * 0.02  # 偏移量为y轴范围的2%
        ax.text(bar.get_x() + bar.get_width() / 2., height + offset,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 美化图表
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # 设置x轴标签旋转，避免重叠
    ax.tick_params(axis='x', rotation=45)

# 调整布局
plt.tight_layout()

# 添加图例（在图表外部）
fig.legend(conditions, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=2, frameon=False, fontsize=12)

# 显示图表
plt.show()

# # 保存图表
# plt.savefig('提示词工程影响分析.png', dpi=300, bbox_inches='tight',
#             facecolor='white', edgecolor='none')
# print("图表已保存为 '提示词工程影响分析.png'")