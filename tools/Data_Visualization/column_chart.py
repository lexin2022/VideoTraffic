<<<<<<< HEAD
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Times New Roman']

labels = ['RF', 'SGD', 'SVC', 'RC', 'KNN', 'DT']
precision = [98.64, 91.16, 90.87, 90.68, 93.79, 98.56]
recall = [99.09, 98.86, 100, 100, 96.58, 95.66]
F1 = [98.86, 94.85, 95.22, 95.11, 95.16, 97.10]
# marks = ["o", "X", "+", "*", "O"]

x = np.arange(len(labels))  # 标签位置
width = 0.2  # 柱状图的宽度

fig, ax = plt.subplots(figsize=(8, 8))
rects1 = ax.bar(x - width, precision, width, label='precision', hatch="++", color='w', edgecolor="k")
rects2 = ax.bar(x + 0.01, recall, width, label='recall', hatch="XX", color='w', edgecolor="k")
rects3 = ax.bar(x + width + 0.02, F1, width, label='F1', hatch="**", color='w', edgecolor="k")

plt.ylim(90, 101)
# 为y轴、标题和x轴等添加一些文本。
ax.set_ylabel('Result(%)', fontsize=20)
ax.set_xlabel('Classification models', fontsize=20)
# ax.set_xlabel('Classification Algorithm', fontsize=16)
# ax.set_title('标题')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
ax.legend(fontsize=18)

plt.tick_params(labelsize=16)

plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 字体设置
#
# precision = [85.25, 89.93, 93.12, 87.22, 88.07]
# recall = [97.92, 97.83, 98.03, 96.36, 97.22]
# Ratio = ['1:8', '1:16', '1:32', '1:64', '1:128']
# # marks = ["o", "X", "+", "*", "O"]
#
# x = np.arange(len(Ratio))  # 标签位置
# width = 0.2  # 柱状图的宽度
#
# fig, ax = plt.subplots(figsize=(8, 8))
# rects1 = ax.bar(x - width, precision, width, label='precision', hatch="++", color='w', edgecolor="k")
# rects2 = ax.bar(x + 0.01, recall, width, label='recall', hatch="XX", color='w', edgecolor="k")
#
#
# plt.ylim(80, 101)
# # 为y轴、标题和x轴等添加一些文本。
# ax.set_ylabel('Result(%)', fontsize=20)
# ax.set_xlabel('Sampling ratio', fontsize=20)
# # ax.set_xlabel('Classification Algorithm', fontsize=16)
# # ax.set_title('标题')
# ax.set_xticks(x)
# ax.set_xticklabels(Ratio, fontsize=16)
# ax.legend(fontsize=18)
#
# plt.tick_params(labelsize=16)
#
=======
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Times New Roman']

labels = ['RF', 'SGD', 'SVC', 'RC', 'KNN', 'DT']
precision = [98.64, 91.16, 90.87, 90.68, 93.79, 98.56]
recall = [99.09, 98.86, 100, 100, 96.58, 95.66]
F1 = [98.86, 94.85, 95.22, 95.11, 95.16, 97.10]
# marks = ["o", "X", "+", "*", "O"]

x = np.arange(len(labels))  # 标签位置
width = 0.2  # 柱状图的宽度

fig, ax = plt.subplots(figsize=(8, 8))
rects1 = ax.bar(x - width, precision, width, label='precision', hatch="++", color='w', edgecolor="k")
rects2 = ax.bar(x + 0.01, recall, width, label='recall', hatch="XX", color='w', edgecolor="k")
rects3 = ax.bar(x + width + 0.02, F1, width, label='F1', hatch="**", color='w', edgecolor="k")

plt.ylim(90, 101)
# 为y轴、标题和x轴等添加一些文本。
ax.set_ylabel('Result(%)', fontsize=20)
ax.set_xlabel('Classification models', fontsize=20)
# ax.set_xlabel('Classification Algorithm', fontsize=16)
# ax.set_title('标题')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
ax.legend(fontsize=18)

plt.tick_params(labelsize=16)

plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 字体设置
#
# precision = [85.25, 89.93, 93.12, 87.22, 88.07]
# recall = [97.92, 97.83, 98.03, 96.36, 97.22]
# Ratio = ['1:8', '1:16', '1:32', '1:64', '1:128']
# # marks = ["o", "X", "+", "*", "O"]
#
# x = np.arange(len(Ratio))  # 标签位置
# width = 0.2  # 柱状图的宽度
#
# fig, ax = plt.subplots(figsize=(8, 8))
# rects1 = ax.bar(x - width, precision, width, label='precision', hatch="++", color='w', edgecolor="k")
# rects2 = ax.bar(x + 0.01, recall, width, label='recall', hatch="XX", color='w', edgecolor="k")
#
#
# plt.ylim(80, 101)
# # 为y轴、标题和x轴等添加一些文本。
# ax.set_ylabel('Result(%)', fontsize=20)
# ax.set_xlabel('Sampling ratio', fontsize=20)
# # ax.set_xlabel('Classification Algorithm', fontsize=16)
# # ax.set_title('标题')
# ax.set_xticks(x)
# ax.set_xticklabels(Ratio, fontsize=16)
# ax.legend(fontsize=18)
#
# plt.tick_params(labelsize=16)
#
>>>>>>> 74d556a (tools)
# plt.show()