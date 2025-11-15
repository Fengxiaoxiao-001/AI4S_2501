import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# 根据您图片中的数据设置
difficulties = ['简单', '中等', '困难']

# 从您提供的原始数据计算提升幅度
# 无提示词工程数据（简单、中等、困难三个难度级别的平均值）
# 请替换为您的实际数据
# 平均创造性得分（无提示词工程）
creativity_no_prompt = [3.868479167
, 4.940958333
, 4.315583333
]  # 示例数据，请替换为实际数据

# 一题多解能力评分（无提示词工程）
diversity_no_prompt = [0.433063709
,0.279662414
,0.328791551
]  # 示例数据，请替换为实际数据

# 输出稳定性得分（无提示词工程）
stability_no_prompt = [0.98125
,0.959375
,0.975
] # 示例数据，请替换为实际数据

# 有提示词工程数据（简单、中等、困难三个难度级别的平均值）
# 请替换为您的实际数据
# 平均创造性得分（有提示词工程）
creativity_with_prompt = [4.3288125
, 5.430708333
, 4.287708333
]  # 示例数据，请替换为实际数据

# 一题多解能力评分（有提示词工程）
diversity_with_prompt =  [0.345749176
, 0.341995108
, 0.466765786
]  # 示例数据，请替换为实际数据

# 输出稳定性得分（有提示词工程）
stability_with_prompt = [0.990625
, 0.9875
, 0.934375
]  # 示例数据，请替换为实际数据

# 计算提升幅度（有提示词工程 - 无提示词工程）
creativity_improvement = [creativity_with_prompt[i] - creativity_no_prompt[i] for i in range(3)]
diversity_improvement = [diversity_with_prompt[i] - diversity_no_prompt[i] for i in range(3)]
stability_improvement = [stability_with_prompt[i] - stability_no_prompt[i] for i in range(3)]

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制折线
plt.plot(difficulties, creativity_improvement, marker='o', linewidth=2, label='平均创造性得分提升', color='green')
plt.plot(difficulties, diversity_improvement, marker='s', linewidth=2, label='一题多解能力评分提升', color='red')
plt.plot(difficulties, stability_improvement, marker='^', linewidth=2, label='输出稳定性得分提升', color='purple')

# 设置标题和标签
plt.title('提示词工程在不同难度下的效果提升分析', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('难度', fontsize=12)
plt.ylabel('性能提升幅度', fontsize=12)

# 设置Y轴范围（根据实际数据调整）
# 计算所有提升值的范围，留出一些边距
all_improvements = creativity_improvement + diversity_improvement + stability_improvement
y_min = min(all_improvements) - 0.1
y_max = max(all_improvements) + 0.1
plt.ylim(y_min, y_max)

# 添加图例
plt.legend(loc='best', fontsize=10)

# 添加网格
plt.grid(True, alpha=0.3)

# 在数据点上添加数值标签
for i, (cr, div, st) in enumerate(zip(creativity_improvement, diversity_improvement, stability_improvement)):
    plt.annotate(f'{cr:.3f}', (difficulties[i], cr), textcoords="offset points", xytext=(0,10), ha='center', fontsize=18)
    plt.annotate(f'{div:.3f}', (difficulties[i], div), textcoords="offset points", xytext=(0,10), ha='center', fontsize=18)
    plt.annotate(f'{st:.3f}', (difficulties[i], st), textcoords="offset points", xytext=(0,10), ha='center', fontsize=18)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()