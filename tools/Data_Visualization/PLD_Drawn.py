<<<<<<< HEAD
# import pandas as pd
# import csv
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np
#
# sns.set(palette="muted", color_codes=True)
# #sns.set_style({"font.sans-serif": ['simhei', 'Droid Sans Fallback']})
# matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
#
# base_path = "E:/video_traffic_datas/"
# base_filename = 'e720 60.pcapng.173.194.22.184.pcap.6,173.194.22.184,443,192.168.20.78,36231,3338176071,_'
# data1 = pd.read_csv(base_path + base_filename + 'In.Pack_Len.csv')
# data2 = pd.read_csv(base_path + base_filename + 'Out.Pack_Len.csv')
#
# da1 = data1['pck_len']
# da2 = data2['pck_len']
#
# fig, ax = plt.subplots()
#
# sns.distplot(da1, hist= True, color='b', label='request packet lengths in ' + str(len(da1)) + ' pcks')
# sns.distplot(da2, hist= True, color='r', label='response packet lengths in ' + str(len(da2)) + ' pcks')
#
# plt.ylabel('PDF(%)')
# plt.xlabel("the transmission data length(Bytes)")
#
# # ax.set_title('QUIC video streaming')
# # ax.set_title('TCP video streaming')
# # ax.set_title('P2P video streaming')
# # ax.set_title('Not video streaming')
# ax.set_title("Picture flow Length probability distribution")
# ax.legend()
#
# png_path = base_path + base_filename + '.png'
# fig.savefig(png_path, dpi = 400)
# plt.show()

# import pandas as pd
#
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib


# if __name__ == "__main__":
#     """
#     利用指定路径的csv文件，画出该流的pdf图
#     """
#     sns.set(palette="muted", color_codes=True)
#     matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
#
#     filepath = 'E:/video_traffic_datas/mix_data/ITP_KNN-Comparative Experiment/Time to reach 2000 packets.xlsx'
#     data = pd.read_excel(filepath)
#     X = data['time cost']
#     fig, ax = plt.subplots()
#     sns.distplot(X, hist=False, color='b', label='Time to reach 2000 packets')
#     # ax.set_yscale('log')  # 将坐标转换成对数
#
#     plt.ylabel('Probability Density')
#     plt.xlabel("Time(s)")
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     ax.legend()
#     # figure_path =str + '.png'
#     # plt.savefig(figure_path)
#     plt.show()

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


if __name__ == "__main__":
    """
    利用指定路径的csv文件，画出该流的pdf图
    """
    sns.set(palette="muted", color_codes=False)
    sns.set_style("white")
    # sns.set_style({"font.sans-serif": ['simhei', 'Droid Sans Fallback']})
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    path1 = 'C:/Users/13137/Desktop/paper/different flow/pdf/chat_flow_01.pck_count.csv'
    path2 = 'C:/Users/13137/Desktop/paper/different flow/pdf/file_flow_01.pck_count.csv'
    path3 = 'C:/Users/13137/Desktop/paper/different flow/pdf/game_flow_01.pck_count.csv'
    path4 = 'C:/Users/13137/Desktop/paper/different flow/pdf/video_flow_01.pck_count.csv'

    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data3 = pd.read_csv(path3)
    data4 = pd.read_csv(path4)

    da1 = data1.iloc[:, 0]
    da2 = data2.iloc[:, 0]
    da3 = data3.iloc[:, 0]
    da4 = data4.iloc[:, 0]

    # da2 = data2['length']
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1600)
    # sns.distplot(da1, hist=False, color='#8B8989', label='chat flow')
    # sns.distplot(da2, hist=False, color= '#FFD700', label='file flow')
    # sns.distplot(da3, hist=False, color='b', label='game flow')
    # sns.distplot(da4, hist=False, color='r', label= 'video flow')
    sns.distplot(da1, hist=False, color='#8B8989', label='chat flow')
    sns.distplot(da2, hist=False, color= '#FFD700', label='file flow')
    sns.distplot(da3, hist=False, color='b', label='game flow')
    sns.distplot(da4, hist=False, color='r', label= 'video flow')
    # sns.distplot(da2, hist=False, color='r', label='request data length')
    # ax.set_yscale('log')  # 将坐标转换成对数



    plt.ylabel('Probability Density', fontsize=24)
    plt.xlabel("Packet Payload Length (Bytes)", fontsize=24)
    ax.legend(fontsize=20)
    # figure_path = str1 + '/figure/' + str2 + '.png'
    # plt.savefig(figure_path)
    plt.tick_params(labelsize=18)

=======
# import pandas as pd
# import csv
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np
#
# sns.set(palette="muted", color_codes=True)
# #sns.set_style({"font.sans-serif": ['simhei', 'Droid Sans Fallback']})
# matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
#
# base_path = "E:/video_traffic_datas/"
# base_filename = 'e720 60.pcapng.173.194.22.184.pcap.6,173.194.22.184,443,192.168.20.78,36231,3338176071,_'
# data1 = pd.read_csv(base_path + base_filename + 'In.Pack_Len.csv')
# data2 = pd.read_csv(base_path + base_filename + 'Out.Pack_Len.csv')
#
# da1 = data1['pck_len']
# da2 = data2['pck_len']
#
# fig, ax = plt.subplots()
#
# sns.distplot(da1, hist= True, color='b', label='request packet lengths in ' + str(len(da1)) + ' pcks')
# sns.distplot(da2, hist= True, color='r', label='response packet lengths in ' + str(len(da2)) + ' pcks')
#
# plt.ylabel('PDF(%)')
# plt.xlabel("the transmission data length(Bytes)")
#
# # ax.set_title('QUIC video streaming')
# # ax.set_title('TCP video streaming')
# # ax.set_title('P2P video streaming')
# # ax.set_title('Not video streaming')
# ax.set_title("Picture flow Length probability distribution")
# ax.legend()
#
# png_path = base_path + base_filename + '.png'
# fig.savefig(png_path, dpi = 400)
# plt.show()

# import pandas as pd
#
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib


# if __name__ == "__main__":
#     """
#     利用指定路径的csv文件，画出该流的pdf图
#     """
#     sns.set(palette="muted", color_codes=True)
#     matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
#
#     filepath = 'E:/video_traffic_datas/mix_data/ITP_KNN-Comparative Experiment/Time to reach 2000 packets.xlsx'
#     data = pd.read_excel(filepath)
#     X = data['time cost']
#     fig, ax = plt.subplots()
#     sns.distplot(X, hist=False, color='b', label='Time to reach 2000 packets')
#     # ax.set_yscale('log')  # 将坐标转换成对数
#
#     plt.ylabel('Probability Density')
#     plt.xlabel("Time(s)")
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     ax.legend()
#     # figure_path =str + '.png'
#     # plt.savefig(figure_path)
#     plt.show()

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


if __name__ == "__main__":
    """
    利用指定路径的csv文件，画出该流的pdf图
    """
    sns.set(palette="muted", color_codes=False)
    sns.set_style("white")
    # sns.set_style({"font.sans-serif": ['simhei', 'Droid Sans Fallback']})
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    path1 = 'C:/Users/13137/Desktop/paper/different flow/pdf/chat_flow_01.pck_count.csv'
    path2 = 'C:/Users/13137/Desktop/paper/different flow/pdf/file_flow_01.pck_count.csv'
    path3 = 'C:/Users/13137/Desktop/paper/different flow/pdf/game_flow_01.pck_count.csv'
    path4 = 'C:/Users/13137/Desktop/paper/different flow/pdf/video_flow_01.pck_count.csv'

    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data3 = pd.read_csv(path3)
    data4 = pd.read_csv(path4)

    da1 = data1.iloc[:, 0]
    da2 = data2.iloc[:, 0]
    da3 = data3.iloc[:, 0]
    da4 = data4.iloc[:, 0]

    # da2 = data2['length']
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1600)
    # sns.distplot(da1, hist=False, color='#8B8989', label='chat flow')
    # sns.distplot(da2, hist=False, color= '#FFD700', label='file flow')
    # sns.distplot(da3, hist=False, color='b', label='game flow')
    # sns.distplot(da4, hist=False, color='r', label= 'video flow')
    sns.distplot(da1, hist=False, color='#8B8989', label='chat flow')
    sns.distplot(da2, hist=False, color= '#FFD700', label='file flow')
    sns.distplot(da3, hist=False, color='b', label='game flow')
    sns.distplot(da4, hist=False, color='r', label= 'video flow')
    # sns.distplot(da2, hist=False, color='r', label='request data length')
    # ax.set_yscale('log')  # 将坐标转换成对数



    plt.ylabel('Probability Density', fontsize=24)
    plt.xlabel("Packet Payload Length (Bytes)", fontsize=24)
    ax.legend(fontsize=20)
    # figure_path = str1 + '/figure/' + str2 + '.png'
    # plt.savefig(figure_path)
    plt.tick_params(labelsize=18)

>>>>>>> 74d556a (tools)
    plt.show()