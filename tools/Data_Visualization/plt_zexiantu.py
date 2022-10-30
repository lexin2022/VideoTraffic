<<<<<<< HEAD
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['Times New Roman']

names = ['100', '250', '500', '1000', '1500', '2000', '3000', '4000', '5000']
numbers = [100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
precision = [0.9567, 0.965, 0.9677, 0.9739, 0.984, 0.9883, 0.985, 0.9869, 0.9875]
recall = [0.9518, 0.9634, 0.9673, 0.9733, 0.9836, 0.9881, 0.9863, 0.9868, 0.9863]
f1_score = [0.9531, 0.9638, 0.9675, 0.9734, 0.9837, 0.9881, 0.9856, 0.9868, 0.9869]
accuracy = [0.9518, 0.9634, 0.9733, 0.9733, 0.9836, 0.9881, 0.9853, 0.9868, 0.986]

precision = [i * 100 for i in precision]
recall = [i * 100 for i in recall]
f1_score = [i * 100 for i in f1_score]
accuracy = [i * 100 for i in accuracy]


plt.figure(figsize=(8, 8))
plt.plot(numbers, precision, marker='o', label='weighted avg precision')
plt.plot(numbers, recall, marker='*', label='weighted avg recall')
plt.plot(numbers, f1_score, marker='^', label='weighted avg f1_score')
plt.plot(numbers, accuracy, marker='1', label='accuracy')

plt.ylim(95, 100)

plt.legend(loc='lower right', fontsize=18)  # 让图例生效
plt.xticks(numbers, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)

plt.xlabel("Extract features from different numbers of packets", fontsize=20)
plt.ylabel("Result(%)", fontsize=20)
# plt.title("Model performance based on early packet features")
plt.tick_params(labelsize=14)

plt.show()

# import matplotlib.pyplot as plt
# from pylab import *                                 #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
#
# names = ['1/8', '1/16', '1/32', '1/64', '1/128']
# numbers = [1, 2, 3, 4, 5]
# precision = [0.8525, 0.8993, 0.9312, 0.8722, 0.8807]
# recall = [0.9792, 0.9783, 0.9803, 0.9636, 0.9722]
#
# plt.plot(numbers, precision, marker='o', label='precision(%)')
# plt.plot(numbers, recall, marker='*', label='recall(%)')
#
# plt.ylim(0.8, 1.0)
#
# plt.legend(loc='lower right')  # 让图例生效
# plt.xticks(numbers, names, rotation=45)
# plt.margins(0)
# plt.subplots_adjust(bottom=0.15)
#
# plt.xlabel("Sampling rate")
# plt.ylabel("Score")
# # plt.title("Model performance based on early packet features")
#
=======
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['Times New Roman']

names = ['100', '250', '500', '1000', '1500', '2000', '3000', '4000', '5000']
numbers = [100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
precision = [0.9567, 0.965, 0.9677, 0.9739, 0.984, 0.9883, 0.985, 0.9869, 0.9875]
recall = [0.9518, 0.9634, 0.9673, 0.9733, 0.9836, 0.9881, 0.9863, 0.9868, 0.9863]
f1_score = [0.9531, 0.9638, 0.9675, 0.9734, 0.9837, 0.9881, 0.9856, 0.9868, 0.9869]
accuracy = [0.9518, 0.9634, 0.9733, 0.9733, 0.9836, 0.9881, 0.9853, 0.9868, 0.986]

precision = [i * 100 for i in precision]
recall = [i * 100 for i in recall]
f1_score = [i * 100 for i in f1_score]
accuracy = [i * 100 for i in accuracy]


plt.figure(figsize=(8, 8))
plt.plot(numbers, precision, marker='o', label='weighted avg precision')
plt.plot(numbers, recall, marker='*', label='weighted avg recall')
plt.plot(numbers, f1_score, marker='^', label='weighted avg f1_score')
plt.plot(numbers, accuracy, marker='1', label='accuracy')

plt.ylim(95, 100)

plt.legend(loc='lower right', fontsize=18)  # 让图例生效
plt.xticks(numbers, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)

plt.xlabel("Extract features from different numbers of packets", fontsize=20)
plt.ylabel("Result(%)", fontsize=20)
# plt.title("Model performance based on early packet features")
plt.tick_params(labelsize=14)

plt.show()

# import matplotlib.pyplot as plt
# from pylab import *                                 #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
#
# names = ['1/8', '1/16', '1/32', '1/64', '1/128']
# numbers = [1, 2, 3, 4, 5]
# precision = [0.8525, 0.8993, 0.9312, 0.8722, 0.8807]
# recall = [0.9792, 0.9783, 0.9803, 0.9636, 0.9722]
#
# plt.plot(numbers, precision, marker='o', label='precision(%)')
# plt.plot(numbers, recall, marker='*', label='recall(%)')
#
# plt.ylim(0.8, 1.0)
#
# plt.legend(loc='lower right')  # 让图例生效
# plt.xticks(numbers, names, rotation=45)
# plt.margins(0)
# plt.subplots_adjust(bottom=0.15)
#
# plt.xlabel("Sampling rate")
# plt.ylabel("Score")
# # plt.title("Model performance based on early packet features")
#
>>>>>>> 74d556a (tools)
# plt.show()