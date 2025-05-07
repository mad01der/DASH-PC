import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['Calibri']

# plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
# plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题



# 需加入cdf原始数据，即cdf_method，对应下面plot代码

visible_percentages = np.array([
    0.9524, 0.9527, 0.9528, 0.9530, 0.9531, 0.9535, 0.9539, 0.9540, 0.9541, 0.9544,
    0.9546, 0.9566, 0.9580, 0.9586, 0.9612, 0.9613, 0.9630, 0.9670, 0.9674, 0.9688,
    0.9699, 0.9719, 0.9723, 0.9728, 0.9730, 0.9734, 0.9741, 0.9757, 0.9770, 0.9781,
    0.9791, 0.9817, 0.9821, 0.9831, 0.9840, 0.9841, 0.9842, 0.9859, 0.9865, 0.9871,
    0.9891, 0.9891, 0.9892, 0.9895, 0.9898, 0.9899, 0.9900, 0.9910, 0.9917, 0.9919,
    0.9926, 0.9930, 0.9932, 0.9940, 0.9942, 0.9945, 0.9945, 0.9953, 0.9955, 0.9967,
    0.9968, 0.9968, 0.9975, 0.9975, 0.9978, 0.9982, 0.9982, 0.9984, 0.9984, 0.9993,
    0.9994, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

sorted_percentages = np.sort(visible_percentages)
cdf = np.arange(1, len(sorted_percentages)+1) / len(sorted_percentages)
x_smooth = np.linspace(sorted_percentages.min(), sorted_percentages.max(), 300)
y_smooth = np.interp(x_smooth, sorted_percentages, cdf)
# plt.figure(figsize=(4, 3))
fig, ax = plt.subplots(figsize=(6.5, 5))
# ax.set_aspect(1)
bbox_props = dict(boxstyle="rarrow", fc="#EEE9E9", ec="#FF0000", lw=3)
t = ax.text(6.2, 0.2, "Better", ha="center", va="center", rotation=0,
            size=22,
            color = "#000000",
            bbox=bbox_props)
# plt.text(2,0.4,"Better")

bb = t.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.35)

plt.plot(x_smooth, y_smooth, color='#FF0000', linewidth=4 ,zorder=1)
# plt.plot(x, cdf_eRB, color='#FF9200', linewidth=4, label='enhanced RB',zorder=1)
# plt.plot(x, cdf_eMPC, color='#CFB99E', linewidth=4, label='enhanced MPC',zorder=1)
# plt.plot(x, cdf_eRLB, color='#66C062', linewidth=4, label='enhanced RLB',zorder=1)
# plt.plot(x, cdf_eTwoTier, color='#5B9BD5', linewidth=4, label='enhanced Twotier',zorder=1)

# plt.plot(x, cdf_BB, color='#FF0000', linewidth=2, linestyle='--',  label='BB',zorder=2)
# plt.plot(x, cdf_RB, color='#FF9200', linewidth=2, linestyle='--', label='RB',zorder=2)
# plt.plot(x, cdf_MPC, color='#CFB99E', linewidth=2, linestyle='--', label='MPC',zorder=2)
# plt.plot(x, cdf_RLB, color='#66C062', linewidth=2, linestyle='--', label='RLB',zorder=2)
# plt.plot(x, cdf_TwoTier, color='#5B9BD5', linewidth=2, linestyle='--', label='Twotier',zorder=2)

plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.ylabel("CDF",fontsize=20)  # Y轴标签
plt.legend()  # 让图例生效
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel('the proportion that falls inside the viewport',labelpad=-3, fontsize=20)  # X轴标签
plt.legend(frameon=False, loc=5, bbox_to_anchor=(0.47,0.65), fontsize=14)
# plt.legend(frameon=False, loc=0, fontsize=18)

from matplotlib.ticker import MultipleLocator
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
x_major_locator=MultipleLocator(1)#以每15显示
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()  #gca:get current axis得到当前轴
#设置图片的右边框和上边框为不显示
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
plt.grid()

plt.show()
