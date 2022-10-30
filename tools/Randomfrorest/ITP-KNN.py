<<<<<<< HEAD
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler  # 归一化函数

import psutil
import os

def train_knn_model(train_x, test_x, train_y, test_y):
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(train_x, train_y)
    return model


def train_from_forward_pcks(dataframe):
    """
    :param dataframe: 数据集
    :return:NONE
    """

    df_features = dataframe.iloc[0:, 13:83]
    # print(df_features.columns)
    df_labels = dataframe['label']

    # 2.1 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.3, random_state=1)
    model = train_knn_model(train_x, test_x, train_y, test_y)  # 训练模型
    print(classification_report(test_y, model.predict(test_x), digits=4))

    return model


def main():
    # 数据集路径
    filename = "E:/video_traffic_datas/mix_data/ITP_KNN-Comparative Experiment/capture_video_flows.pcap.comp.PAI.csv"

    dataframe = pd.read_csv(filename)
    train_from_forward_pcks(dataframe)


if __name__ == "__main__":
    main()
=======
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler  # 归一化函数

import psutil
import os

def train_knn_model(train_x, test_x, train_y, test_y):
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(train_x, train_y)
    return model


def train_from_forward_pcks(dataframe):
    """
    :param dataframe: 数据集
    :return:NONE
    """

    df_features = dataframe.iloc[0:, 13:83]
    # print(df_features.columns)
    df_labels = dataframe['label']

    # 2.1 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.3, random_state=1)
    model = train_knn_model(train_x, test_x, train_y, test_y)  # 训练模型
    print(classification_report(test_y, model.predict(test_x), digits=4))

    return model


def main():
    # 数据集路径
    filename = "E:/video_traffic_datas/mix_data/ITP_KNN-Comparative Experiment/capture_video_flows.pcap.comp.PAI.csv"

    dataframe = pd.read_csv(filename)
    train_from_forward_pcks(dataframe)


if __name__ == "__main__":
    main()
>>>>>>> 74d556a (tools)
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)