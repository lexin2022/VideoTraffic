<<<<<<< HEAD
import pandas as pd
import numpy as np
import csv
import joblib
import matplotlib.pyplot as plt
import time
import os
import pydotplus    # 画图工具，用于绘制决策树图像
from lazypredict.Supervised import LazyClassifier  # sklearn 批量模型拟合

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.tree import export_text, export_graphviz
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc  # 计算roc和auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler    # 归一化函数
from sklearn.inspection import permutation_importance   # 特征重要性
from sklearn.model_selection import GridSearchCV


# 设置print()打印显示的参数
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def train_rf_clf_model(train_x, test_x, train_y, test_y):
    model = RandomForestClassifier(n_estimators=118, random_state=0)
    model.fit(train_x, train_y)
    print(classification_report(test_y, model.predict(test_x)))
    return model
    # # 网格搜索最佳参数组合
    # model = RandomForestClassifier(random_state=0, n_jobs=-1)
    # params = {
    #     'n_estimators': range(100, 120, 1),
    # }
    # grid = GridSearchCV(model, param_grid=params, n_jobs=-1)
    # grid.fit(train_x, train_y)
    # model = grid.best_params_
    # print(type(grid.best_params_))
    # print(grid.best_params_)
    # return model


def train_ab_clf_model(train_x, test_x, train_y, test_y):    # 集成学习模型训练
    model = AdaBoostClassifier(n_estimators=300, random_state=1, learning_rate=0.1)
    model.fit(train_x, train_y)
    print(classification_report(test_y, model.predict(test_x)))
    return model

    # # 网格搜索最佳参数组合
    # model = AdaBoostClassifier(random_state=0)
    # params = {
    #     'n_estimators': range(100, 1000, 100),
    #     'learning_rate': np.arange(0.1, 1, 0.1)
    # }
    # grid = GridSearchCV(model, param_grid=params, n_jobs=-1)
    # grid.fit(train_x, train_y)
    # model = grid.best_params_
    # print(type(grid.best_params_))
    # print(grid.best_params_)
    # return model


def plot_roc(labels, predict_label):
    fpr, tpr, threshold = roc_curve(labels, predict_label)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    # plt.savefig('figures/PC5.png') #将ROC图片进行保存
    plt.show()


def plot_pr(labels, predict_label):
    precision, recall, thresholds = precision_recall_curve(labels, predict_label)
    lw = 2
    plt.figure(figsize=(10, 10))

    plt.plot(precision, recall, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R curve')
    plt.show()


def plot_importance_mdi(model, feature_names):
    # Feature importance based on mean decrease in impurity
    importances = model.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=importances, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


def save_to_csv(path, title, list):
    """
    写如csv文件
    :param path: 写入路径
    :param title: csv文件标题
    :param list: csv文件数据
    :return:
    """
    data_frame = pd.DataFrame(columns=title, data=list)
    data_frame.to_csv(path, index=0)    # 不保存行索引


def main_train(data_path):
    """
    :param data_path: 统计流中pck=500时的特征，csv文件路径
    :return:
    """
    dfOrigin_data = pd.read_csv(data_path)
    df_features = dfOrigin_data[['Max', 'Q3', 'Mid', 'Q1', 'Min']]
    df_labels = dfOrigin_data['label']

    # 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # # 先比较不同分类模型的得分
    # clf_model = LazyClassifier(custom_metric=classification_report, predictions=True)
    # models, predictions = clf_model.fit(train_x, test_x, train_y, test_y)
    # models.to_csv("classifier_comparison_report_pulse_-1_220112.csv", index_label="index_label")

    model = train_ab_clf_model(train_x, test_x, train_y, test_y)
    # model = train_rf_clf_model(train_x, test_x, train_y, test_y)


def main():
    # 数据集路径
    filename = "E:/video_traffic_datas/mix_data/mix_videos_images.csv"

    # 训练模型，区分视频和图床
    main_train(filename)

    # 4 保存模型
    # joblib.dump(rf_clf_model, "./models/rf_clf_1000_v3.pkl")

    # 5 结果可视化
    # plot_roc(df_labels, predict_label)
    # plot_pr(df_labels, predict_label)


if __name__ == "__main__":
    main()
=======
import pandas as pd
import numpy as np
import csv
import joblib
import matplotlib.pyplot as plt
import time
import os
import pydotplus    # 画图工具，用于绘制决策树图像
from lazypredict.Supervised import LazyClassifier  # sklearn 批量模型拟合

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.tree import export_text, export_graphviz
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc  # 计算roc和auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler    # 归一化函数
from sklearn.inspection import permutation_importance   # 特征重要性
from sklearn.model_selection import GridSearchCV


# 设置print()打印显示的参数
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def train_rf_clf_model(train_x, test_x, train_y, test_y):
    model = RandomForestClassifier(n_estimators=118, random_state=0)
    model.fit(train_x, train_y)
    print(classification_report(test_y, model.predict(test_x)))
    return model
    # # 网格搜索最佳参数组合
    # model = RandomForestClassifier(random_state=0, n_jobs=-1)
    # params = {
    #     'n_estimators': range(100, 120, 1),
    # }
    # grid = GridSearchCV(model, param_grid=params, n_jobs=-1)
    # grid.fit(train_x, train_y)
    # model = grid.best_params_
    # print(type(grid.best_params_))
    # print(grid.best_params_)
    # return model


def train_ab_clf_model(train_x, test_x, train_y, test_y):    # 集成学习模型训练
    model = AdaBoostClassifier(n_estimators=300, random_state=1, learning_rate=0.1)
    model.fit(train_x, train_y)
    print(classification_report(test_y, model.predict(test_x)))
    return model

    # # 网格搜索最佳参数组合
    # model = AdaBoostClassifier(random_state=0)
    # params = {
    #     'n_estimators': range(100, 1000, 100),
    #     'learning_rate': np.arange(0.1, 1, 0.1)
    # }
    # grid = GridSearchCV(model, param_grid=params, n_jobs=-1)
    # grid.fit(train_x, train_y)
    # model = grid.best_params_
    # print(type(grid.best_params_))
    # print(grid.best_params_)
    # return model


def plot_roc(labels, predict_label):
    fpr, tpr, threshold = roc_curve(labels, predict_label)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    # plt.savefig('figures/PC5.png') #将ROC图片进行保存
    plt.show()


def plot_pr(labels, predict_label):
    precision, recall, thresholds = precision_recall_curve(labels, predict_label)
    lw = 2
    plt.figure(figsize=(10, 10))

    plt.plot(precision, recall, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R curve')
    plt.show()


def plot_importance_mdi(model, feature_names):
    # Feature importance based on mean decrease in impurity
    importances = model.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=importances, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


def save_to_csv(path, title, list):
    """
    写如csv文件
    :param path: 写入路径
    :param title: csv文件标题
    :param list: csv文件数据
    :return:
    """
    data_frame = pd.DataFrame(columns=title, data=list)
    data_frame.to_csv(path, index=0)    # 不保存行索引


def main_train(data_path):
    """
    :param data_path: 统计流中pck=500时的特征，csv文件路径
    :return:
    """
    dfOrigin_data = pd.read_csv(data_path)
    df_features = dfOrigin_data[['Max', 'Q3', 'Mid', 'Q1', 'Min']]
    df_labels = dfOrigin_data['label']

    # 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # # 先比较不同分类模型的得分
    # clf_model = LazyClassifier(custom_metric=classification_report, predictions=True)
    # models, predictions = clf_model.fit(train_x, test_x, train_y, test_y)
    # models.to_csv("classifier_comparison_report_pulse_-1_220112.csv", index_label="index_label")

    model = train_ab_clf_model(train_x, test_x, train_y, test_y)
    # model = train_rf_clf_model(train_x, test_x, train_y, test_y)


def main():
    # 数据集路径
    filename = "E:/video_traffic_datas/mix_data/mix_videos_images.csv"

    # 训练模型，区分视频和图床
    main_train(filename)

    # 4 保存模型
    # joblib.dump(rf_clf_model, "./models/rf_clf_1000_v3.pkl")

    # 5 结果可视化
    # plot_roc(df_labels, predict_label)
    # plot_pr(df_labels, predict_label)


if __name__ == "__main__":
    main()
>>>>>>> 74d556a (tools)
