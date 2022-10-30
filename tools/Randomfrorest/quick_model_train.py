<<<<<<< HEAD
import pandas as pd
import numpy as np
import csv
import joblib
import matplotlib.pyplot as plt
import time
import os
import pydotplus  # 画图工具，用于绘制决策树图像
from pandas import DataFrame as df
from lazypredict.Supervised import LazyClassifier  # sklearn 批量模型拟合

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.tree import export_text, export_graphviz
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc  # 计算roc和auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler  # 归一化函数
from sklearn.inspection import permutation_importance  # 特征重要性
from sklearn.model_selection import GridSearchCV

import psutil
import os

# # 设置print()打印显示的参数
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def train_rf_clf_model(train_x, test_x, train_y, test_y):
    # model = RandomForestClassifier(n_estimators=500, max_features=4, random_state=1)
    model = RandomForestClassifier(n_estimators=500, max_depth=17, random_state=1)
    model.fit(train_x, train_y)
    return model

    # # 网格搜索最佳参数组合
    # params = {
    #     'n_estimators': range(100, 2001, 100),
    #     # 'max_features': range(3, 8, 1),
    #     'max_depth': range(3, 20, 1),
    #     # 'min_samples_split': range(10, 101, 10),
    #     # 'min_samples_leaf': range(5, 51, 5)
    # }
    # grid = GridSearchCV(estimator=RandomForestClassifier(max_features=4, oob_score=True, random_state=1),
    #                     param_grid=params, n_jobs=-1, scoring='f1', cv=10)
    # grid.fit(train_x, train_y)
    #
    # means = grid.cv_results_['mean_test_score']
    # std = grid.cv_results_['std_test_score']
    # params = grid.cv_results_['params']
    # for mean, std, param in zip(means, std, params):
    #     print("mean : %f std : %f %r" % (mean, std, param))
    # print('best_params :', grid.best_params_)


def train_ab_clf_model(train_x, test_x, train_y, test_y):  # 集成学习模型训练
    model = AdaBoostClassifier(n_estimators=200, random_state=1, learning_rate=0.4)
    model.fit(train_x, train_y)
    return model

    # # 网格搜索最佳参数组合
    # model = AdaBoostClassifier(random_state=1)
    # params = {
    #     'n_estimators': range(100, 1000, 100),
    #     'learning_rate': np.arange(0.1, 1, 0.1)
    # }
    # grid = GridSearchCV(model, param_grid=params, n_jobs=-1)
    # grid.fit(train_x, train_y)
    # params = grid.best_params_
    # # print(type(grid.best_params_))
    # print(params)
    # return model


def plot_roc(y_true, probas_pred_y):
    fpr, tpr, threshold = roc_curve(y_true, probas_pred_y)  # 计算真正率和假正率
    roc_auc = auc(fpr, tpr)  # 计算auc的值

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, lw=1, color='red', alpha=0.9,
             label="RFC roc curve, AUC=%0.3f" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC curve', fontsize=12)
    plt.legend(loc="lower right")
    plt.style.use('seaborn-darkgrid')
    # plt.savefig('figures/PC5.png') #将ROC图片进行保存
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

    plt.figure(figsize=(10, 10))
    plt.title('P-R curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])

    plt.plot(precision, recall, color='darkorange')
    plt.show()

    # return thresholds[idx]


def plot_importance_mdi(model, feature_names):
    # # Feature importance based on mean decrease in impurity
    # importances = model.feature_importances_
    # # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    # forest_importances = pd.Series(importances, index=feature_names)
    #
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=importances, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # plt.show()

    features_import = df(feature_names, columns=['feature'])
    features_import['importance'] = model.feature_importances_  # 默认按照gini计算特征重要性
    features_import.sort_values('importance', inplace=True)
    features_import = features_import.iloc[26:, :]
    print(features_import)

    # 绘图
    plt.figure(figsize=(16, 9))
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文黑体
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plt.rcParams['axes.unicode_minus'] = False # 负值显示
    plt.barh(features_import['feature'], features_import['importance'], height=0.7, hatch="/", color='lemonchiffon',
             edgecolor='black')  # 更多颜色可参见颜色大全
    plt.xlabel('Feature importance', fontsize=24)  # x 轴
    # plt.ylabel('features', fontsize=20)  # y轴
    # plt.title('Feature Importances')  # 标题

    plt.xlim(0, 0.19)
    plt.xticks(fontsize=20)
    plt.yticks(rotation=45, fontsize=20)

    for a, b in zip(features_import['importance'], features_import['feature']):  # 添加数字标签
        # print(a, b)
        plt.text(a + 0.001, b, '%.3f' % float(a), size=18)  # a+0.001代表标签位置在柱形图上方0.001处
    plt.show()


def print_decision_tree_image(tree, out_path):
    # 将决策树绘制成图到你当前的路径
    # 加入Graphviz的环境路径
    os.environ["PATH"] += os.pathsep + "E:/Graphviz/bin"

    # 绘图并导出
    dot_data = export_graphviz(tree, out_file=None, feature_names=df_features.columns,
                               filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.get_nodes()[7].set_fillcolor("#FFF2DD")
    if os.path.exists(out_path):
        pass
    else:
        graph.write_png(out_path)  # 当前文件夹生成out.png


def save_to_csv(path, title, list):
    """
    写如csv文件
    :param path: 写入路径
    :param title: csv文件标题
    :param list: csv文件数据
    :return:
    """
    data_frame = pd.DataFrame(columns=title, data=list)
    data_frame.to_csv(path, index=0)  # 不保存行索引


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name, 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        for data in datas:
            writer.writerow(data)  # writerow()方法是一行一行写入.
    # print("保存文件成功，处理结束")


def get_best_threshold(y_true, probas_pred):
    """
    :param y_true: 真实标签
    :param probas_pred: 标签被标记为1的概率
    :return: recall=1时，F1最大时的阈值
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, probas_pred)
    # F1 = 2 * precisions * recalls / (precisions + recalls)
    tp = fp = fn = 0
    max_F1 = 0
    mask_index = 0
    # print(recalls)
    # print(thresholds)

    for index in range(len(thresholds)):
        cur_F1 = 2 * precisions[index] * recalls[index] / (precisions[index] + recalls[index])
        if recalls[index] > 0.995 and cur_F1 > max_F1:
            max_F1 = cur_F1
            mask_index = index
    print("p={0}, r={1}, F1={2}".format(precisions[mask_index], recalls[mask_index], max_F1))
    return thresholds[mask_index]


def train_from_forward_pcks(dataframe):
    """
    :param dataframe: 数据集
    :return:NONE
    """
    # dfOrigin_data = dataframe
    # df_features = dfOrigin_data[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl']]
    # df_labels = dfOrigin_data['label']
    # X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    # Y_labels = np.array(df_labels, dtype='float32')
    # train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    df_features = dataframe.iloc[0:, 9:55]
    df_features.drop(['o_data_p', 'i_data_p', 'o_len', 'i_len', 'o_data_l', 'i_data_l', 'begin TM', 'end TM'], axis=1, inplace=True)
    # df_features = dataframe[['per_i_(0)', 'r_o_dp', 'r_o_dl', 'per_o_(>1300)', 'r_o_len', 'per_i_(1-100)',
    #                          'o_spd_len',
    #                          'r_o_pck']]  # , 'per_o_(0)', 'i_spd_len']]    #, 'o_spd_pck', 'i_spd_pck', 'per_o_(1-100)']]
    # print(df_features.columns)
    df_labels = dataframe['label']

    # 2.1 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.3, random_state=1)

    # # 先比较不同分类模型的得分
    # clf_model = LazyClassifier(custom_metric=classification_report, predictions=True)
    # models, predictions = clf_model.fit(train_x, test_x, train_y, test_y)
    # models.to_csv("classifier_comparison_report_3000.csv", index_label="index_label")

    model = train_rf_clf_model(train_x, test_x, train_y, test_y)  # 训练模型
    # model = train_ab_clf_model(train_x, test_x, train_y, test_y)
    # threshold = get_best_threshold(test_y, model.predict_proba(test_x)[:, 1])
    # print("best threshold = {0}".format(threshold))
    # predict_y = (model.predict_proba(test_x)[:, 1] >= threshold).astype(int)

    # 模型评估
    # plot_roc(test_y, model.predict_proba(test_x)[:, 1])

    # # 输出预测结果
    # predict_Y = model.predict(X_features)
    # prob_positive_examples = model.predict_proba(X_features)[:, 1]
    # dataframe['probability'] = prob_positive_examples
    # dataframe['predict'] = predict_Y
    # correct = []
    # for i in range(len(predict_Y)):
    #     result = 'T'
    #     origin_label, predict_label = Y_labels[i], predict_Y[i]
    #     if origin_label != predict_label:
    #         if predict_label == 1:
    #             result = 'Fp'
    #         if predict_label == 0:
    #             result = 'FN'
    #     correct.append(result)
    # dataframe['correct'] = correct
    # # dataframe.to_excel('E:/video_traffic_datas/mix_data/video_data_cap_mix/predict_prob_2000_pcks_0.xlsx', index=None)

    # 输出特征重要性
    # df_import = pd.DataFrame(features_importance, index=None)
    # df_import = df_import.T
    # df_import.columns = df_features.columns
    # df_import = pd.DataFrame(columns=df_features.columns, data=(model.feature_importances_).resize((1, len(model.feature_importances_))))
    # df_import.to_csv('E:/video_traffic_datas/mix_data/video_data_cap_mix/feature_importance_2000.csv', index=None)
    plot_importance_mdi(model, df_features.columns)

    ##### 1、feature_importances_（适用于决策树、随机森林、GBDT、xgboost、lightgbm）
    # # 重要性
    # features_import = df(df_features.columns, columns=['feature'])
    # features_import['importance'] = model.feature_importances_  # 默认按照gini计算特征重要性
    # features_import.sort_values('importance', inplace=True)
    # # 绘图
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文黑体
    # # plt.rcParams['axes.unicode_minus'] = False # 负值显示
    # plt.barh(features_import['feature'], features_import['importance'], height=0.7, color='#008792',
    #          edgecolor='#005344')  # 更多颜色可参见颜色大全
    # plt.xlabel('feature importance')  # x 轴
    # plt.ylabel('features')  # y轴
    # plt.title('Feature Importances')  # 标题
    # for a, b in zip(features_import['importance'], features_import['feature']):  # 添加数字标签
    #     print(a, b)
    #     plt.text(a + 0.001, b, '%.3f' % float(a))  # a+0.001代表标签位置在柱形图上方0.001处
    # plt.show()

    # print(classification_report(test_y, model.predict(test_x), digits=4))

    return model


def main():
    # 数据集路径
    filename_100 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.100.csv"
    filename_250 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.250.csv"
    filename_500 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.500.csv"
    filename_1000 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.1000.csv"
    filename_1500 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.1500.csv"
    filename_2000 = "E:/video_traffic_datas/mix_data/capture data bak/capture_video_flows.pcap.flow.stat.2000.csv"
    filename_3000 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.3000.csv"
    filename_4000 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.4000.csv"
    filename_5000 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.5000.csv"

    # df_100 = pd.read_csv(filename_100)
    # train_from_forward_pcks(df_100)

    # df_250 = pd.read_csv(filename_250)
    # train_from_forward_pcks(df_250)

    # df_500 = pd.read_csv(filename_500)
    # train_from_forward_pcks(df_500)

    # df_1000 = pd.read_csv(filename_1000)
    # train_from_forward_pcks(df_1000)

    # df_1500 = pd.read_csv(filename_1500)
    # train_from_forward_pcks(df_1500)

    df_2000 = pd.read_csv(filename_2000)
    model = train_from_forward_pcks(df_2000)

    # df_3000 = pd.read_csv(filename_3000)
    # train_from_forward_pcks(df_3000)

    # df_4000 = pd.read_csv(filename_4000)
    # train_from_forward_pcks(df_4000)

    # df_5000 = pd.read_csv(filename_5000)
    # train_from_forward_pcks(df_5000)

    # 4 保存模型
    # joblib.dump(model, "./models/model.f1_per9886.pkl")

    # 5 结果可视化
    # plot_roc(df_labels, predict_label)
    # plot_pr(df_labels, predict_label)
    # plot_importance_MDI(rf_clf_model, df_features.columns)
    # print_decision_tree_image(rf_clf_model.estimators_[0], 'decision_tree.png')


if __name__ == "__main__":
    main()
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
=======
import pandas as pd
import numpy as np
import csv
import joblib
import matplotlib.pyplot as plt
import time
import os
import pydotplus  # 画图工具，用于绘制决策树图像
from pandas import DataFrame as df
from lazypredict.Supervised import LazyClassifier  # sklearn 批量模型拟合

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.tree import export_text, export_graphviz
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc  # 计算roc和auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler  # 归一化函数
from sklearn.inspection import permutation_importance  # 特征重要性
from sklearn.model_selection import GridSearchCV

import psutil
import os

# # 设置print()打印显示的参数
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def train_rf_clf_model(train_x, test_x, train_y, test_y):
    # model = RandomForestClassifier(n_estimators=500, max_features=4, random_state=1)
    model = RandomForestClassifier(n_estimators=500, max_depth=17, random_state=1)
    model.fit(train_x, train_y)
    return model

    # # 网格搜索最佳参数组合
    # params = {
    #     'n_estimators': range(100, 2001, 100),
    #     # 'max_features': range(3, 8, 1),
    #     'max_depth': range(3, 20, 1),
    #     # 'min_samples_split': range(10, 101, 10),
    #     # 'min_samples_leaf': range(5, 51, 5)
    # }
    # grid = GridSearchCV(estimator=RandomForestClassifier(max_features=4, oob_score=True, random_state=1),
    #                     param_grid=params, n_jobs=-1, scoring='f1', cv=10)
    # grid.fit(train_x, train_y)
    #
    # means = grid.cv_results_['mean_test_score']
    # std = grid.cv_results_['std_test_score']
    # params = grid.cv_results_['params']
    # for mean, std, param in zip(means, std, params):
    #     print("mean : %f std : %f %r" % (mean, std, param))
    # print('best_params :', grid.best_params_)


def train_ab_clf_model(train_x, test_x, train_y, test_y):  # 集成学习模型训练
    model = AdaBoostClassifier(n_estimators=200, random_state=1, learning_rate=0.4)
    model.fit(train_x, train_y)
    return model

    # # 网格搜索最佳参数组合
    # model = AdaBoostClassifier(random_state=1)
    # params = {
    #     'n_estimators': range(100, 1000, 100),
    #     'learning_rate': np.arange(0.1, 1, 0.1)
    # }
    # grid = GridSearchCV(model, param_grid=params, n_jobs=-1)
    # grid.fit(train_x, train_y)
    # params = grid.best_params_
    # # print(type(grid.best_params_))
    # print(params)
    # return model


def plot_roc(y_true, probas_pred_y):
    fpr, tpr, threshold = roc_curve(y_true, probas_pred_y)  # 计算真正率和假正率
    roc_auc = auc(fpr, tpr)  # 计算auc的值

    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, lw=1, color='red', alpha=0.9,
             label="RFC roc curve, AUC=%0.3f" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC curve', fontsize=12)
    plt.legend(loc="lower right")
    plt.style.use('seaborn-darkgrid')
    # plt.savefig('figures/PC5.png') #将ROC图片进行保存
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

    plt.figure(figsize=(10, 10))
    plt.title('P-R curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])

    plt.plot(precision, recall, color='darkorange')
    plt.show()

    # return thresholds[idx]


def plot_importance_mdi(model, feature_names):
    # # Feature importance based on mean decrease in impurity
    # importances = model.feature_importances_
    # # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    # forest_importances = pd.Series(importances, index=feature_names)
    #
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=importances, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # plt.show()

    features_import = df(feature_names, columns=['feature'])
    features_import['importance'] = model.feature_importances_  # 默认按照gini计算特征重要性
    features_import.sort_values('importance', inplace=True)
    features_import = features_import.iloc[26:, :]
    print(features_import)

    # 绘图
    plt.figure(figsize=(16, 9))
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文黑体
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plt.rcParams['axes.unicode_minus'] = False # 负值显示
    plt.barh(features_import['feature'], features_import['importance'], height=0.7, hatch="/", color='lemonchiffon',
             edgecolor='black')  # 更多颜色可参见颜色大全
    plt.xlabel('Feature importance', fontsize=24)  # x 轴
    # plt.ylabel('features', fontsize=20)  # y轴
    # plt.title('Feature Importances')  # 标题

    plt.xlim(0, 0.19)
    plt.xticks(fontsize=20)
    plt.yticks(rotation=45, fontsize=20)

    for a, b in zip(features_import['importance'], features_import['feature']):  # 添加数字标签
        # print(a, b)
        plt.text(a + 0.001, b, '%.3f' % float(a), size=18)  # a+0.001代表标签位置在柱形图上方0.001处
    plt.show()


def print_decision_tree_image(tree, out_path):
    # 将决策树绘制成图到你当前的路径
    # 加入Graphviz的环境路径
    os.environ["PATH"] += os.pathsep + "E:/Graphviz/bin"

    # 绘图并导出
    dot_data = export_graphviz(tree, out_file=None, feature_names=df_features.columns,
                               filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.get_nodes()[7].set_fillcolor("#FFF2DD")
    if os.path.exists(out_path):
        pass
    else:
        graph.write_png(out_path)  # 当前文件夹生成out.png


def save_to_csv(path, title, list):
    """
    写如csv文件
    :param path: 写入路径
    :param title: csv文件标题
    :param list: csv文件数据
    :return:
    """
    data_frame = pd.DataFrame(columns=title, data=list)
    data_frame.to_csv(path, index=0)  # 不保存行索引


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name, 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        for data in datas:
            writer.writerow(data)  # writerow()方法是一行一行写入.
    # print("保存文件成功，处理结束")


def get_best_threshold(y_true, probas_pred):
    """
    :param y_true: 真实标签
    :param probas_pred: 标签被标记为1的概率
    :return: recall=1时，F1最大时的阈值
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, probas_pred)
    # F1 = 2 * precisions * recalls / (precisions + recalls)
    tp = fp = fn = 0
    max_F1 = 0
    mask_index = 0
    # print(recalls)
    # print(thresholds)

    for index in range(len(thresholds)):
        cur_F1 = 2 * precisions[index] * recalls[index] / (precisions[index] + recalls[index])
        if recalls[index] > 0.995 and cur_F1 > max_F1:
            max_F1 = cur_F1
            mask_index = index
    print("p={0}, r={1}, F1={2}".format(precisions[mask_index], recalls[mask_index], max_F1))
    return thresholds[mask_index]


def train_from_forward_pcks(dataframe):
    """
    :param dataframe: 数据集
    :return:NONE
    """
    # dfOrigin_data = dataframe
    # df_features = dfOrigin_data[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl']]
    # df_labels = dfOrigin_data['label']
    # X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    # Y_labels = np.array(df_labels, dtype='float32')
    # train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    df_features = dataframe.iloc[0:, 9:55]
    df_features.drop(['o_data_p', 'i_data_p', 'o_len', 'i_len', 'o_data_l', 'i_data_l', 'begin TM', 'end TM'], axis=1, inplace=True)
    # df_features = dataframe[['per_i_(0)', 'r_o_dp', 'r_o_dl', 'per_o_(>1300)', 'r_o_len', 'per_i_(1-100)',
    #                          'o_spd_len',
    #                          'r_o_pck']]  # , 'per_o_(0)', 'i_spd_len']]    #, 'o_spd_pck', 'i_spd_pck', 'per_o_(1-100)']]
    # print(df_features.columns)
    df_labels = dataframe['label']

    # 2.1 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.3, random_state=1)

    # # 先比较不同分类模型的得分
    # clf_model = LazyClassifier(custom_metric=classification_report, predictions=True)
    # models, predictions = clf_model.fit(train_x, test_x, train_y, test_y)
    # models.to_csv("classifier_comparison_report_3000.csv", index_label="index_label")

    model = train_rf_clf_model(train_x, test_x, train_y, test_y)  # 训练模型
    # model = train_ab_clf_model(train_x, test_x, train_y, test_y)
    # threshold = get_best_threshold(test_y, model.predict_proba(test_x)[:, 1])
    # print("best threshold = {0}".format(threshold))
    # predict_y = (model.predict_proba(test_x)[:, 1] >= threshold).astype(int)

    # 模型评估
    # plot_roc(test_y, model.predict_proba(test_x)[:, 1])

    # # 输出预测结果
    # predict_Y = model.predict(X_features)
    # prob_positive_examples = model.predict_proba(X_features)[:, 1]
    # dataframe['probability'] = prob_positive_examples
    # dataframe['predict'] = predict_Y
    # correct = []
    # for i in range(len(predict_Y)):
    #     result = 'T'
    #     origin_label, predict_label = Y_labels[i], predict_Y[i]
    #     if origin_label != predict_label:
    #         if predict_label == 1:
    #             result = 'Fp'
    #         if predict_label == 0:
    #             result = 'FN'
    #     correct.append(result)
    # dataframe['correct'] = correct
    # # dataframe.to_excel('E:/video_traffic_datas/mix_data/video_data_cap_mix/predict_prob_2000_pcks_0.xlsx', index=None)

    # 输出特征重要性
    # df_import = pd.DataFrame(features_importance, index=None)
    # df_import = df_import.T
    # df_import.columns = df_features.columns
    # df_import = pd.DataFrame(columns=df_features.columns, data=(model.feature_importances_).resize((1, len(model.feature_importances_))))
    # df_import.to_csv('E:/video_traffic_datas/mix_data/video_data_cap_mix/feature_importance_2000.csv', index=None)
    plot_importance_mdi(model, df_features.columns)

    ##### 1、feature_importances_（适用于决策树、随机森林、GBDT、xgboost、lightgbm）
    # # 重要性
    # features_import = df(df_features.columns, columns=['feature'])
    # features_import['importance'] = model.feature_importances_  # 默认按照gini计算特征重要性
    # features_import.sort_values('importance', inplace=True)
    # # 绘图
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文黑体
    # # plt.rcParams['axes.unicode_minus'] = False # 负值显示
    # plt.barh(features_import['feature'], features_import['importance'], height=0.7, color='#008792',
    #          edgecolor='#005344')  # 更多颜色可参见颜色大全
    # plt.xlabel('feature importance')  # x 轴
    # plt.ylabel('features')  # y轴
    # plt.title('Feature Importances')  # 标题
    # for a, b in zip(features_import['importance'], features_import['feature']):  # 添加数字标签
    #     print(a, b)
    #     plt.text(a + 0.001, b, '%.3f' % float(a))  # a+0.001代表标签位置在柱形图上方0.001处
    # plt.show()

    # print(classification_report(test_y, model.predict(test_x), digits=4))

    return model


def main():
    # 数据集路径
    filename_100 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.100.csv"
    filename_250 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.250.csv"
    filename_500 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.500.csv"
    filename_1000 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.1000.csv"
    filename_1500 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.1500.csv"
    filename_2000 = "E:/video_traffic_datas/mix_data/capture data bak/capture_video_flows.pcap.flow.stat.2000.csv"
    filename_3000 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.3000.csv"
    filename_4000 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.4000.csv"
    filename_5000 = "E:/video_traffic_datas/mix_data/capture_video_flows.pcap.flow.stat.5000.csv"

    # df_100 = pd.read_csv(filename_100)
    # train_from_forward_pcks(df_100)

    # df_250 = pd.read_csv(filename_250)
    # train_from_forward_pcks(df_250)

    # df_500 = pd.read_csv(filename_500)
    # train_from_forward_pcks(df_500)

    # df_1000 = pd.read_csv(filename_1000)
    # train_from_forward_pcks(df_1000)

    # df_1500 = pd.read_csv(filename_1500)
    # train_from_forward_pcks(df_1500)

    df_2000 = pd.read_csv(filename_2000)
    model = train_from_forward_pcks(df_2000)

    # df_3000 = pd.read_csv(filename_3000)
    # train_from_forward_pcks(df_3000)

    # df_4000 = pd.read_csv(filename_4000)
    # train_from_forward_pcks(df_4000)

    # df_5000 = pd.read_csv(filename_5000)
    # train_from_forward_pcks(df_5000)

    # 4 保存模型
    # joblib.dump(model, "./models/model.f1_per9886.pkl")

    # 5 结果可视化
    # plot_roc(df_labels, predict_label)
    # plot_pr(df_labels, predict_label)
    # plot_importance_MDI(rf_clf_model, df_features.columns)
    # print_decision_tree_image(rf_clf_model.estimators_[0], 'decision_tree.png')


if __name__ == "__main__":
    main()
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
>>>>>>> 74d556a (tools)
