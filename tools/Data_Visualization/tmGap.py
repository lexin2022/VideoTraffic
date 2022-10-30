<<<<<<< HEAD
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
mpl.rcParams['font.sans-serif'] = ['SimHei']

data = pd.read_excel('C:/Users/13137/Desktop/paper/different flow/tmGap/video_rrtv_01_tmGap.xlsx', sheet_name='Sheet3')

X1 = data['Forward Link']
X2 = data['Backward Link']
Y1 = data['F_label']
Y2 = data['B_label']

plt.xlim([0, 40])
plt.ylim([-1.5, 1.5])
plt.scatter(X1, Y1, label='Forward Link', s=1)
plt.scatter(X2, Y2, label='Backward Link', s=1)

# plt.legend(loc='lower right')  # 让图例生效

plt.xlabel("Times(s)", fontsize=20)
plt.yticks([-0.5, 0.5], ['Forward Link', 'Backward Link'], rotation=90, fontsize=20)
plt.title("Packet Arrival Time", fontsize=25)

plt.show()

# data = pd.read_excel('C:/Users/13137/Desktop/paper/different flow/IO ratio/video_flow_02_packets.xlsx')
#
# X = data['Interval start']
# Y = data['All packets']
#
# plt.xlim([0, 12])
# # plt.ylim([-1.5, 1.5])
# plt.plot(X, Y)
# plt.plot(4.7, 2035, marker='x')
#
# # plt.legend(loc='lower right')  # 让图例生效
#
# plt.xlabel("Times(s)", fontsize=15)
# plt.ylabel("Packets number", fontsize=15)
# plt.title("Number of packets over time", fontsize=18)
#
=======
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
mpl.rcParams['font.sans-serif'] = ['SimHei']

data = pd.read_excel('C:/Users/13137/Desktop/paper/different flow/tmGap/video_rrtv_01_tmGap.xlsx', sheet_name='Sheet3')

X1 = data['Forward Link']
X2 = data['Backward Link']
Y1 = data['F_label']
Y2 = data['B_label']

plt.xlim([0, 40])
plt.ylim([-1.5, 1.5])
plt.scatter(X1, Y1, label='Forward Link', s=1)
plt.scatter(X2, Y2, label='Backward Link', s=1)

# plt.legend(loc='lower right')  # 让图例生效

plt.xlabel("Times(s)", fontsize=20)
plt.yticks([-0.5, 0.5], ['Forward Link', 'Backward Link'], rotation=90, fontsize=20)
plt.title("Packet Arrival Time", fontsize=25)

plt.show()

# data = pd.read_excel('C:/Users/13137/Desktop/paper/different flow/IO ratio/video_flow_02_packets.xlsx')
#
# X = data['Interval start']
# Y = data['All packets']
#
# plt.xlim([0, 12])
# # plt.ylim([-1.5, 1.5])
# plt.plot(X, Y)
# plt.plot(4.7, 2035, marker='x')
#
# # plt.legend(loc='lower right')  # 让图例生效
#
# plt.xlabel("Times(s)", fontsize=15)
# plt.ylabel("Packets number", fontsize=15)
# plt.title("Number of packets over time", fontsize=18)
#
>>>>>>> 74d556a (tools)
# plt.show()