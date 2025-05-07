import numpy as np
import matplotlib.pyplot as plt

# 原始数据
rate_no_vhp = [
[
    "56.16",
    "56.40",
    "-9.28",
    "20.81",
    "21.70",
    "24.08",
    "-10.50",
    "5.98",
    "5.52",
    "5.68",
    "5.87",
    "4.69",
    "5.55",
    "5.28",
    "5.31",
    "5.47",
    "5.64",
    "6.03",
    "6.37",
    "6.15",
    "6.06",
    "-11.82",
    "4.05",
    "2.61",
    "3.04",
    "3.44",
    "4.37",
    "4.17",
    "4.65",
    "4.11",
    "9.79",
    "-25.06",
    "10.59",
    "-30.34",
    "-0.72",
    "1.05",
    "1.51",
    "1.30",
    "1.66",
    "0.87"
],
[
    "56.16",
    "56.40",
    "-9.28",
    "20.81",
    "21.70",
    "24.08",
    "-10.50",
    "5.98",
    "5.52",
    "5.68",
    "5.87",
    "4.69",
    "5.55",
    "5.28",
    "5.31",
    "5.47",
    "5.64",
    "6.03",
    "6.37",
    "6.15",
    "6.06",
    "-11.82",
    "4.05",
    "2.61",
    "3.04",
    "3.44",
    "4.37",
    "4.17",
    "4.65",
    "4.11",
    "4.64",
    "5.35",
    "3.61",
    "2.53",
    "-0.72",
    "1.05",
    "1.51",
    "1.30",
    "1.66",
    "0.87"
],
[
    "56.16",
    "56.40",
    "53.98",
    "-8.22",
    "21.70",
    "-9.65",
    "5.78",
    "5.98",
    "5.52",
    "5.68",
    "5.87",
    "4.69",
    "5.55",
    "7.84",
    "21.24",
    "21.87",
    "22.55",
    "24.11",
    "25.48",
    "24.62",
    "24.26",
    "22.49",
    "-9.49",
    "5.65",
    "7.89",
    "23.44",
    "24.37",
    "24.17",
    "24.65",
    "24.11",
    "24.64",
    "25.35",
    "23.61",
    "22.53",
    "19.28",
    "21.05",
    "21.51",
    "21.30",
    "21.66",
    "20.87"
]
]

rate_vhp = [
[
    "22.46",
    "22.56",
    "21.59",
    "20.81",
    "21.70",
    "24.08",
    "23.11",
    "23.91",
    "22.09",
    "22.71",
    "23.48",
    "18.74",
    "22.20",
    "21.12",
    "21.24",
    "21.87",
    "22.55",
    "24.11",
    "25.48",
    "24.62",
    "24.26",
    "22.49",
    "24.05",
    "22.61",
    "23.04",
    "23.44",
    "24.37",
    "24.17",
    "29.81",
    "60.27",
    "-5.74",
    "25.35",
    "30.59",
    "-10.34",
    "19.28",
    "21.05",
    "21.51",
    "21.30",
    "-6.67",
    "8.10"
],
[
    "22.46",
    "22.56",
    "21.59",
    "20.81",
    "21.70",
    "24.08",
    "23.11",
    "23.91",
    "22.09",
    "22.71",
    "23.48",
    "18.74",
    "22.20",
    "21.12",
    "21.24",
    "21.87",
    "22.55",
    "24.11",
    "25.48",
    "24.62",
    "24.26",
    "22.49",
    "24.05",
    "22.61",
    "23.04",
    "23.44",
    "24.37",
    "24.17",
    "24.65",
    "24.11",
    "24.64",
    "25.35",
    "23.61",
    "22.53",
    "19.28",
    "21.05",
    "21.51",
    "21.30",
    "-6.67",
    "8.10"
],
[
    "22.46",
    "22.56",
    "21.59",
    "20.81",
    "21.70",
    "24.08",
    "23.11",
    "23.91",
    "22.09",
    "22.71",
    "-9.12",
    "4.69",
    "5.55",
    "7.84",
    "21.24",
    "21.87",
    "22.55",
    "24.11",
    "25.48",
    "24.62",
    "24.26",
    "22.49",
    "24.05",
    "22.61",
    "23.04",
    "23.44",
    "24.37",
    "24.17",
    "29.81",
    "60.27",
    "61.61",
    "-5.06",
    "23.61",
    "22.53",
    "19.28",
    "21.05",
    "21.51",
    "21.30",
    "-6.67",
    "8.10"
]
]

# 数据转换
no_vhp_data = [float(item) for sublist in rate_no_vhp for item in sublist]
vhp_data = [float(item) for sublist in rate_vhp for item in sublist]

# 计算统计量
def calculate_stats(data, name):
    avg = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    print(f"{name} Statistics:")
    print(f"  Average: {avg:.2f}")
    print(f"  Variance: {var:.2f}")
    print(f"  Standard Deviation: {std:.2f}\n")
    return avg, var, std

avg_no_vhp, var_no_vhp, std_no_vhp = calculate_stats(no_vhp_data, "No VHP")
avg_vhp, var_vhp, std_vhp = calculate_stats(vhp_data, "VHP")

# 创建CDF函数
def create_cdf(data):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    return sorted_data, yvals

x_no_vhp, y_no_vhp = create_cdf(no_vhp_data)
x_vhp, y_vhp = create_cdf(vhp_data)

# 绘制CDF图
plt.figure(figsize=(12, 6))
plt.plot(x_no_vhp, y_no_vhp, label=f'No VHP (Avg: {avg_no_vhp:.2f}, Var: {var_no_vhp:.2f})', linewidth=2.5)
plt.plot(x_vhp, y_vhp, label=f'VHP (Avg: {avg_vhp:.2f}, Var: {var_vhp:.2f})', linewidth=2.5)

# 添加均值参考线
plt.axvline(avg_no_vhp, color='blue', linestyle=':', alpha=0.7)
plt.axvline(avg_vhp, color='orange', linestyle=':', alpha=0.7)

# 图表装饰
plt.title('CDF Comparison of QoE with Variance Analysis', fontsize=14, pad=20)
plt.xlabel('QoE Value', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, which='both', linestyle='--', alpha=0.4)

# 显示负值区域
plt.xlim(min(min(no_vhp_data), min(vhp_data)) - 5, max(max(no_vhp_data), max(vhp_data)) + 5)

plt.tight_layout()
plt.show()