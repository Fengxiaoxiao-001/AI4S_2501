import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei']  # 或使用其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置数据
labels = ['数学概念理解能力', '逻辑推理能力', '符号操作能力', '问题求解与应用能力']
old_data = [0.3,
8.533333333,
0.3,
4.333333333
]
new_data = [0.3,
7.533333333,
0.3,
4.333333333,
]

# 计算角度
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# 初始化图表
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# 绘制数据
old_data += old_data[:1]
new_data += new_data[:1]

ax.plot(angles, old_data, linewidth=2, linestyle='solid', label='无提示词工程', color='#666666')
ax.fill(angles, old_data, alpha=0.25, color='#666666')

ax.plot(angles, new_data, linewidth=2, linestyle='solid', label='有提示词工程', color='#26A69A')
ax.fill(angles, new_data, alpha=0.25, color='#26A69A')

# 设置标签和标题
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 10)
ax.set_title('Model 7 四维雷达图', size=16, pad=20)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()