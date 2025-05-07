import numpy as np
import matplotlib.pyplot as plt

# 方法和数据
methods = ['Buffer', 'Two-Tier', 'MPC', 'RL-based', 'Rate']
# avg_no_vhr = [21.03, 37.34, 26.60, 30.15, 17.96]
# var_no_vhr = [454.12, 837.23, 815.47, 495.56, 132.47]
# avg_vhr = [26.29, 48.24, 39.14, 38.49, 26.77]
# var_vhr = [489.57, 592.29, 821.35, 473.58, 523.39]


avg_no_vhr = [2.103, 3.734, 2.660, 3.015, 1.796]
var_no_vhr = [4.5412, 8.3723, 8.1547, 4.9556, 1.3247]
avg_vhr = [2.629, 4.824, 3.914, 3.849, 2.677]
var_vhr = [4.8957, 5.9229, 8.2135, 4.7358, 5.2339]
# 标准差
std_no_vhr = np.sqrt(var_no_vhr)
std_vhr = np.sqrt(var_vhr)

# 鲜艳颜色配置
colors_no_vhr = ['#FF6666', '#FFCC00', '#66CC66', '#6699FF', '#FF99FF']
colors_vhr = ['#CC0000', '#FF9900', '#339933', '#3366CC', '#CC33CC']

# 柱状图参数
x = np.arange(len(methods))
width = 0.35

# 绘制
fig, ax = plt.subplots(figsize=(8, 4.5))

for i in range(len(methods)):
    ax.bar(x[i] - width/2, avg_no_vhr[i], width, yerr=std_no_vhr[i], capsize=3,
           color=colors_no_vhr[i], label='No VHR' if i == 0 else "", hatch='//', edgecolor='black', linewidth=0.7)
    ax.bar(x[i] + width/2, avg_vhr[i], width, yerr=std_vhr[i], capsize=3,
           color=colors_vhr[i], label='VHR' if i == 0 else "", edgecolor='black', linewidth=0.7)

ax.set_ylabel('Average MOS', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12)
ax.set_ylim(0, 8)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=12, frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()
