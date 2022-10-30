<<<<<<< HEAD
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
mpl.rcParams['font.sans-serif'] = ['Times New Roman']

data = pd.read_excel('C:/Users/13137/Desktop/paper/different flow/IO ratio/IO_campare.xlsx')

X = data['Interval start']
Y1 = data['video_r_o']
Y2 = data['game_r_o']
Y3 = data['chat_r_o']

plt.figure(figsize=(12, 4))

plt.xlim([0, 200])
plt.ylim([0.2, 1.2])
plt.plot(X, Y1, label='video flow')
plt.plot(X, Y2, label='game flow')
plt.plot(X, Y3, label='chat flow')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='right', fontsize=20)  # 让图例生效

plt.xlabel("Times(s)", fontsize=24)
plt.ylabel("Ratio", fontsize=24)

plt.tick_params(labelsize=18)
# plt.title("The proportion of packets with payload received by the flow in the first 200s")

plt.show()
=======
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
mpl.rcParams['font.sans-serif'] = ['Times New Roman']

data = pd.read_excel('C:/Users/13137/Desktop/paper/different flow/IO ratio/IO_campare.xlsx')

X = data['Interval start']
Y1 = data['video_r_o']
Y2 = data['game_r_o']
Y3 = data['chat_r_o']

plt.figure(figsize=(12, 4))

plt.xlim([0, 200])
plt.ylim([0.2, 1.2])
plt.plot(X, Y1, label='video flow')
plt.plot(X, Y2, label='game flow')
plt.plot(X, Y3, label='chat flow')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='right', fontsize=20)  # 让图例生效

plt.xlabel("Times(s)", fontsize=24)
plt.ylabel("Ratio", fontsize=24)

plt.tick_params(labelsize=18)
# plt.title("The proportion of packets with payload received by the flow in the first 200s")

plt.show()
>>>>>>> 74d556a (tools)
