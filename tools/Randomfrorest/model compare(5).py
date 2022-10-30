<<<<<<< HEAD
import pandas as pd
import numpy as np
import csv
import joblib

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import time
import os
import pydotplus  # 画图工具，用于绘制决策树图像
from lazypredict.Supervised import LazyClassifier  # sklearn 批量模型拟合

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.tree import export_text, export_graphviz
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc  # 计算roc和auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler  # 归一化函数
from sklearn.inspection import permutation_importance  # 特征重要性
from sklearn.model_selection import GridSearchCV


def train_diferent_model(train_x, test_x, train_y, test_y):
    # rfc
    rfc_model = RandomForestClassifier(n_estimators=500, max_depth=17, random_state=1)
    rfc_model.fit(train_x, train_y)
    print(classification_report(test_y, rfc_model.predict(test_x), digits=4))
    rfc_fpr, rfc_tpr, threshold = roc_curve(test_y, rfc_model.predict_proba(test_x)[:, 1])
    rfc_roc_auc = auc(rfc_fpr, rfc_tpr)  # 计算auc的值

    # SGDClassifier
    sgd_model = SGDClassifier(random_state=1)
    sgd_model.fit(train_x, train_y)
    print(classification_report(test_y, sgd_model.predict(test_x), digits=4))
    sgd_fpr, sgd_tpr, threshold = roc_curve(test_y, sgd_model.decision_function(test_x))
    sgd_roc_auc = auc(sgd_fpr, sgd_tpr)  # 计算auc的值

    # SVC
    svc_model = SVC(random_state=1)
    svc_model.fit(train_x, train_y)
    print(classification_report(test_y, svc_model.predict(test_x), digits=4))
    svc_fpr, svc_tpr, threshold = roc_curve(test_y, svc_model.decision_function(test_x))
    svc_roc_auc = auc(svc_fpr, svc_tpr)  # 计算auc的值

    # RidgeClassifier
    rc_model = RidgeClassifier(random_state=1)
    rc_model.fit(train_x, train_y)
    print(classification_report(test_y, rc_model.predict(test_x), digits=4))
    rc_fpr, rc_tpr, threshold = roc_curve(test_y, rc_model.decision_function(test_x))
    rc_roc_auc = auc(rc_fpr, rc_tpr)  # 计算auc的值

    # KNeighborsClassifier
    knc_model = KNeighborsClassifier(n_neighbors=9)
    knc_model.fit(train_x, train_y)
    print(classification_report(test_y, knc_model.predict(test_x), digits=4))
    knc_fpr, knc_tpr, threshold = roc_curve(test_y, knc_model.predict_proba(test_x)[:, 1])
    knc_roc_auc = auc(knc_fpr, knc_tpr)  # 计算auc的值

    # DecisionTreeClassifier
    dtc_model = tree.DecisionTreeClassifier(random_state=1)
    dtc_model.fit(train_x, train_y)
    print(classification_report(test_y, dtc_model.predict(test_x), digits=4))
    dtc_fpr, dtc_tpr, threshold = roc_curve(test_y, dtc_model.predict_proba(test_x)[:, 1])
    dtc_roc_auc = auc(dtc_fpr, dtc_tpr)  # 计算auc的值

    # plt roc
    # plt.style.use('seaborn')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 字体设置
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(rfc_fpr, rfc_tpr, lw=1, color='r', alpha=0.9, label="RFC roc curve, AUC=%0.3f" % rfc_roc_auc)
    ax.plot(sgd_fpr, sgd_tpr, lw=1, color='b', alpha=0.9, label="SGD roc curve, AUC=%0.3f" % sgd_roc_auc)
    ax.plot(svc_fpr, svc_tpr, lw=1, color='c', alpha=0.9, label="SVC roc curve, AUC=%0.3f" % svc_roc_auc)
    ax.plot(rc_fpr, rc_tpr, lw=1, color='k', alpha=0.9, label="RC roc curve, AUC=%0.3f" % rc_roc_auc)
    ax.plot(knc_fpr, knc_tpr, lw=1, color='y', alpha=0.9, label="KNN roc curve, AUC=%0.3f" % knc_roc_auc)
    ax.plot(dtc_fpr, dtc_tpr, lw=1, color='m', alpha=0.9, label="DT roc curve, AUC=%0.3f" % dtc_roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    ax.legend(loc="lower right", fontsize=18)   # 添加图例

    # 嵌入局部放大图
    axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                       bbox_to_anchor=(0.15, 0.6, 0.8, 0.8),
                       bbox_transform=ax.transAxes)
    # 在子图中绘制原始数据
    axins.plot(rfc_fpr, rfc_tpr, lw=1, color='r', alpha=0.9)
    axins.plot(sgd_fpr, sgd_tpr, lw=1, color='b', alpha=0.9)
    axins.plot(svc_fpr, svc_tpr, lw=1, color='c', alpha=0.9)
    axins.plot(rc_fpr, rc_tpr, lw=1, color='k', alpha=0.9)
    axins.plot(knc_fpr, knc_tpr, lw=1, color='y', alpha=0.9)
    axins.plot(dtc_fpr, dtc_tpr, lw=1, color='m', alpha=0.9)

    # 调整子坐标系的显示范围
    axins.set_xlim(0.001, 0.08)
    axins.set_ylim(0.9, 1.01)

    # 建立父坐标系与子坐标系的连接线
    # loc1 loc2: 坐标系的四个角
    # 1 (右上) 2 (左上) 3(左下) 4(右下)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    # ax.set_title('ROC curve', fontsize=16)

    ax.tick_params(labelsize=16)
    axins.tick_params(labelsize=14)

    plt.show()


def plot_pr(y_true, probas_pred):
    """
    :param y_true: 真实标签
    :param probas_pred: 正类的预测概率或决策函数
    :return:基于F1值，输出最佳阈值
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
    # F1 = 2 * precision * recall / (precision + recall)
    # idx = np.argmax(F1)

    plt.figure(figsize=(8, 8))
    # plt.title('P-R curve')
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])

    plt.plot(precision, recall, color='darkorange')
    plt.show()

    # return thresholds[idx]


def train_from_forward_pcks(dataframe):
    """
    :param dataframe: 数据集
    :return:NONE
    """
    df_features = dataframe[['per_i_(0)', 'r_o_dp', 'r_o_dl', 'per_o_(>1300)', 'r_o_len', 'per_i_(1-100)',
                             'o_spd_len', 'r_o_pck']]
    df_labels = dataframe['label']

    # 2.1 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.3, random_state=1)
    train_diferent_model(train_x, test_x, train_y, test_y)  # 训练模型

    # # 模型评估
    # plot_roc(test_y, model.predict_proba(test_x)[:, 1])
    # print(classification_report(test_y, model.predict(test_x), digits=4))


def main():
    # 数据集路径
    filename_2000 = "E:/video_traffic_datas/mix_data/video_data_cap_mix/capture_video_flows.pcap.flow.stat.2000.csv"

    df_2000 = pd.read_csv(filename_2000)
    train_from_forward_pcks(df_2000)


if __name__ == "__main__":
    main()
=======
import pandas as pd
import numpy as np
import csv
import joblib

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import time
import os
import pydotplus  # 画图工具，用于绘制决策树图像
from lazypredict.Supervised import LazyClassifier  # sklearn 批量模型拟合

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.tree import export_text, export_graphviz
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc  # 计算roc和auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler  # 归一化函数
from sklearn.inspection import permutation_importance  # 特征重要性
from sklearn.model_selection import GridSearchCV


def train_diferent_model(train_x, test_x, train_y, test_y):
    # rfc
    rfc_model = RandomForestClassifier(n_estimators=500, max_depth=17, random_state=1)
    rfc_model.fit(train_x, train_y)
    print(classification_report(test_y, rfc_model.predict(test_x), digits=4))
    rfc_fpr, rfc_tpr, threshold = roc_curve(test_y, rfc_model.predict_proba(test_x)[:, 1])
    rfc_roc_auc = auc(rfc_fpr, rfc_tpr)  # 计算auc的值

    # SGDClassifier
    sgd_model = SGDClassifier(random_state=1)
    sgd_model.fit(train_x, train_y)
    print(classification_report(test_y, sgd_model.predict(test_x), digits=4))
    sgd_fpr, sgd_tpr, threshold = roc_curve(test_y, sgd_model.decision_function(test_x))
    sgd_roc_auc = auc(sgd_fpr, sgd_tpr)  # 计算auc的值

    # SVC
    svc_model = SVC(random_state=1)
    svc_model.fit(train_x, train_y)
    print(classification_report(test_y, svc_model.predict(test_x), digits=4))
    svc_fpr, svc_tpr, threshold = roc_curve(test_y, svc_model.decision_function(test_x))
    svc_roc_auc = auc(svc_fpr, svc_tpr)  # 计算auc的值

    # RidgeClassifier
    rc_model = RidgeClassifier(random_state=1)
    rc_model.fit(train_x, train_y)
    print(classification_report(test_y, rc_model.predict(test_x), digits=4))
    rc_fpr, rc_tpr, threshold = roc_curve(test_y, rc_model.decision_function(test_x))
    rc_roc_auc = auc(rc_fpr, rc_tpr)  # 计算auc的值

    # KNeighborsClassifier
    knc_model = KNeighborsClassifier(n_neighbors=9)
    knc_model.fit(train_x, train_y)
    print(classification_report(test_y, knc_model.predict(test_x), digits=4))
    knc_fpr, knc_tpr, threshold = roc_curve(test_y, knc_model.predict_proba(test_x)[:, 1])
    knc_roc_auc = auc(knc_fpr, knc_tpr)  # 计算auc的值

    # DecisionTreeClassifier
    dtc_model = tree.DecisionTreeClassifier(random_state=1)
    dtc_model.fit(train_x, train_y)
    print(classification_report(test_y, dtc_model.predict(test_x), digits=4))
    dtc_fpr, dtc_tpr, threshold = roc_curve(test_y, dtc_model.predict_proba(test_x)[:, 1])
    dtc_roc_auc = auc(dtc_fpr, dtc_tpr)  # 计算auc的值

    # plt roc
    # plt.style.use('seaborn')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 字体设置
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(rfc_fpr, rfc_tpr, lw=1, color='r', alpha=0.9, label="RFC roc curve, AUC=%0.3f" % rfc_roc_auc)
    ax.plot(sgd_fpr, sgd_tpr, lw=1, color='b', alpha=0.9, label="SGD roc curve, AUC=%0.3f" % sgd_roc_auc)
    ax.plot(svc_fpr, svc_tpr, lw=1, color='c', alpha=0.9, label="SVC roc curve, AUC=%0.3f" % svc_roc_auc)
    ax.plot(rc_fpr, rc_tpr, lw=1, color='k', alpha=0.9, label="RC roc curve, AUC=%0.3f" % rc_roc_auc)
    ax.plot(knc_fpr, knc_tpr, lw=1, color='y', alpha=0.9, label="KNN roc curve, AUC=%0.3f" % knc_roc_auc)
    ax.plot(dtc_fpr, dtc_tpr, lw=1, color='m', alpha=0.9, label="DT roc curve, AUC=%0.3f" % dtc_roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    ax.legend(loc="lower right", fontsize=18)   # 添加图例

    # 嵌入局部放大图
    axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                       bbox_to_anchor=(0.15, 0.6, 0.8, 0.8),
                       bbox_transform=ax.transAxes)
    # 在子图中绘制原始数据
    axins.plot(rfc_fpr, rfc_tpr, lw=1, color='r', alpha=0.9)
    axins.plot(sgd_fpr, sgd_tpr, lw=1, color='b', alpha=0.9)
    axins.plot(svc_fpr, svc_tpr, lw=1, color='c', alpha=0.9)
    axins.plot(rc_fpr, rc_tpr, lw=1, color='k', alpha=0.9)
    axins.plot(knc_fpr, knc_tpr, lw=1, color='y', alpha=0.9)
    axins.plot(dtc_fpr, dtc_tpr, lw=1, color='m', alpha=0.9)

    # 调整子坐标系的显示范围
    axins.set_xlim(0.001, 0.08)
    axins.set_ylim(0.9, 1.01)

    # 建立父坐标系与子坐标系的连接线
    # loc1 loc2: 坐标系的四个角
    # 1 (右上) 2 (左上) 3(左下) 4(右下)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=20)
    ax.set_ylabel('True Positive Rate', fontsize=20)
    # ax.set_title('ROC curve', fontsize=16)

    ax.tick_params(labelsize=16)
    axins.tick_params(labelsize=14)

    plt.show()


def plot_pr(y_true, probas_pred):
    """
    :param y_true: 真实标签
    :param probas_pred: 正类的预测概率或决策函数
    :return:基于F1值，输出最佳阈值
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
    # F1 = 2 * precision * recall / (precision + recall)
    # idx = np.argmax(F1)

    plt.figure(figsize=(8, 8))
    # plt.title('P-R curve')
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])

    plt.plot(precision, recall, color='darkorange')
    plt.show()

    # return thresholds[idx]


def train_from_forward_pcks(dataframe):
    """
    :param dataframe: 数据集
    :return:NONE
    """
    df_features = dataframe[['per_i_(0)', 'r_o_dp', 'r_o_dl', 'per_o_(>1300)', 'r_o_len', 'per_i_(1-100)',
                             'o_spd_len', 'r_o_pck']]
    df_labels = dataframe['label']

    # 2.1 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.3, random_state=1)
    train_diferent_model(train_x, test_x, train_y, test_y)  # 训练模型

    # # 模型评估
    # plot_roc(test_y, model.predict_proba(test_x)[:, 1])
    # print(classification_report(test_y, model.predict(test_x), digits=4))


def main():
    # 数据集路径
    filename_2000 = "E:/video_traffic_datas/mix_data/video_data_cap_mix/capture_video_flows.pcap.flow.stat.2000.csv"

    df_2000 = pd.read_csv(filename_2000)
    train_from_forward_pcks(df_2000)


if __name__ == "__main__":
    main()
>>>>>>> 74d556a (tools)
