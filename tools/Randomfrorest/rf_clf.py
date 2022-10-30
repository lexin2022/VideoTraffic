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
# from sklearn.model_selection import GridSearchCV


# # 设置print()打印显示的参数
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def train_rf_clf_model(train_x, test_x, train_y, test_y):
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train_x, train_y)
    return model


def train_ab_clf_model(train_x, test_x, train_y, test_y):    # 集成学习模型训练
    model = AdaBoostClassifier(n_estimators=200, random_state=1, learning_rate=0.9)
    model.fit(train_x, train_y)
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


def plot_roc(y_true, probas_pred):
    fpr, tpr, threshold = roc_curve(y_true, probas_pred)  ###计算真正率和假正率
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


def print_decision_tree_image(tree, out_path):
    # 将决策树绘制成图到你当前的路径
    # 加入Graphviz的环境路径
    os.environ["PATH"] += os.pathsep + "E:/Graphviz/bin"

    # 绘图并导出
    dot_data = export_graphviz(tree, out_file=None, feature_names=df_features.columns,
                               filled = True, rounded = True, special_characters = True)
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
    data_frame.to_csv(path, index=0)    # 不保存行索引


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


def main_stat_500(data_path):
    """
    :param data_path: 统计流中pck=500时的特征，csv文件路径
    :return:
    """
    dfOrigin_data = pd.read_csv(data_path)
    df_features = dfOrigin_data[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl']]
    df_labels = dfOrigin_data['label']
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # # 先比较不同分类模型的得分
    # clf_model = LazyClassifier(custom_metric=classification_report, predictions=True)
    # models, predictions = clf_model.fit(train_x, test_x, train_y, test_y)
    # models.to_csv("classifier_comparison_report_500.csv", index_label="index_label")

    # model = train_rf_clf_model(train_x, test_x, train_y, test_y)    # 训练模型
    model = train_ab_clf_model(train_x, test_x, train_y, test_y)
    threshold = get_best_threshold(test_y, model.predict_proba(test_x)[:, 1])
    print("best threshold = {0}".format(threshold))
    predict_y = (model.predict_proba(test_x)[:, 1] >= threshold).astype(int)

    print(classification_report(test_y, predict_y))


def main_stat_2000(data_path):
    """
    :param data_path: 统计流中pck=2000时的特征，csv文件路径
    :return:
    """
    dfOrigin_data = pd.read_csv(data_path)
    df_features = dfOrigin_data[['o_spd_pck', 'i_spd_pck', 'o_spd_len', 'i_spd_len']]
    df_labels = dfOrigin_data['label']

    # 2.1 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)
    # train_rf_clf_model(train_x, test_x, train_y, test_y)    # 训练模型
    model = train_ab_clf_model(train_x, test_x, train_y, test_y)
    # plot_importance_mdi(model, df_features.columns)
    threshold = get_best_threshold(test_y, model.predict_proba(test_x)[:, 1])
    print("best threshold = {0}".format(threshold))
    predict_y = (model.predict_proba(test_x)[:, 1] >= threshold).astype(int)

    print(classification_report(test_y, predict_y))

def main_stat_5000(data_path):
    """
    :param data_path: 统计流中pck=5000时的特征，csv文件路径
    :return:
    """
    dfOrigin_data = pd.read_csv(data_path)
    df_features = dfOrigin_data.iloc[0:, 25:55]
    df_labels = dfOrigin_data['label']

    # 2.1 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)
    # print(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # rf_model = train_rf_clf_model(train_x, test_x, train_y, test_y)    # 训练模型
    model = train_ab_clf_model(train_x, test_x, train_y, test_y)
    # plot_importance_mdi(ab_model, df_features_part2.columns)
    # threshold = get_best_threshold(test_y, model.predict_proba(test_x)[:, 1])
    # print("best threshold = {0}".format(threshold))
    # predict_y = (model.predict_proba(test_x)[:, 1] >= threshold).astype(int)
    predict_y = model.predict(test_x)

    print(classification_report(test_y, predict_y))


def main():
    # 数据集路径
    filename_500 = "E:/video_traffic_datas/mix_data/Early_training_data_set.pcap.Mixed_rseed-22.pcap.flow.stat.500.csv"
    filename_2000 = "E:/video_traffic_datas/mix_data/Early_training_data_set.pcap.Mixed_rseed-22.pcap.flow.stat.2000.csv"
    filename_5000 = "E:/video_traffic_datas/mix_data/Early_training_data_set.pcap.Mixed_rseed-22.pcap.flow.stat.5000.csv"

    # 分别训练三级模型
    # main_stat_500(filename_500)
    # main_stat_2000(filename_2000)
    rf_clf_model = main_stat_5000(filename_5000)

    # 4 保存模型
    # joblib.dump(rf_clf_model, "./models/rf_clf_5000_1226.pkl")

    # 5 结果可视化
    # plot_roc(df_labels, predict_label)
    # plot_pr(df_labels, predict_label)
    # plot_importance_MDI(rf_clf_model, df_features.columns)
    # print_decision_tree_image(rf_clf_model.estimators_[0], 'decision_tree.png')


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
# from sklearn.model_selection import GridSearchCV


# # 设置print()打印显示的参数
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def train_rf_clf_model(train_x, test_x, train_y, test_y):
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train_x, train_y)
    return model


def train_ab_clf_model(train_x, test_x, train_y, test_y):    # 集成学习模型训练
    model = AdaBoostClassifier(n_estimators=200, random_state=1, learning_rate=0.9)
    model.fit(train_x, train_y)
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


def plot_roc(y_true, probas_pred):
    fpr, tpr, threshold = roc_curve(y_true, probas_pred)  ###计算真正率和假正率
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


def print_decision_tree_image(tree, out_path):
    # 将决策树绘制成图到你当前的路径
    # 加入Graphviz的环境路径
    os.environ["PATH"] += os.pathsep + "E:/Graphviz/bin"

    # 绘图并导出
    dot_data = export_graphviz(tree, out_file=None, feature_names=df_features.columns,
                               filled = True, rounded = True, special_characters = True)
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
    data_frame.to_csv(path, index=0)    # 不保存行索引


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


def main_stat_500(data_path):
    """
    :param data_path: 统计流中pck=500时的特征，csv文件路径
    :return:
    """
    dfOrigin_data = pd.read_csv(data_path)
    df_features = dfOrigin_data[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl']]
    df_labels = dfOrigin_data['label']
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # # 先比较不同分类模型的得分
    # clf_model = LazyClassifier(custom_metric=classification_report, predictions=True)
    # models, predictions = clf_model.fit(train_x, test_x, train_y, test_y)
    # models.to_csv("classifier_comparison_report_500.csv", index_label="index_label")

    # model = train_rf_clf_model(train_x, test_x, train_y, test_y)    # 训练模型
    model = train_ab_clf_model(train_x, test_x, train_y, test_y)
    threshold = get_best_threshold(test_y, model.predict_proba(test_x)[:, 1])
    print("best threshold = {0}".format(threshold))
    predict_y = (model.predict_proba(test_x)[:, 1] >= threshold).astype(int)

    print(classification_report(test_y, predict_y))


def main_stat_2000(data_path):
    """
    :param data_path: 统计流中pck=2000时的特征，csv文件路径
    :return:
    """
    dfOrigin_data = pd.read_csv(data_path)
    df_features = dfOrigin_data[['o_spd_pck', 'i_spd_pck', 'o_spd_len', 'i_spd_len']]
    df_labels = dfOrigin_data['label']

    # 2.1 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)
    # train_rf_clf_model(train_x, test_x, train_y, test_y)    # 训练模型
    model = train_ab_clf_model(train_x, test_x, train_y, test_y)
    # plot_importance_mdi(model, df_features.columns)
    threshold = get_best_threshold(test_y, model.predict_proba(test_x)[:, 1])
    print("best threshold = {0}".format(threshold))
    predict_y = (model.predict_proba(test_x)[:, 1] >= threshold).astype(int)

    print(classification_report(test_y, predict_y))

def main_stat_5000(data_path):
    """
    :param data_path: 统计流中pck=5000时的特征，csv文件路径
    :return:
    """
    dfOrigin_data = pd.read_csv(data_path)
    df_features = dfOrigin_data.iloc[0:, 25:55]
    df_labels = dfOrigin_data['label']

    # 2.1 特征均值方差归一化
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)
    # print(X_features)

    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # rf_model = train_rf_clf_model(train_x, test_x, train_y, test_y)    # 训练模型
    model = train_ab_clf_model(train_x, test_x, train_y, test_y)
    # plot_importance_mdi(ab_model, df_features_part2.columns)
    # threshold = get_best_threshold(test_y, model.predict_proba(test_x)[:, 1])
    # print("best threshold = {0}".format(threshold))
    # predict_y = (model.predict_proba(test_x)[:, 1] >= threshold).astype(int)
    predict_y = model.predict(test_x)

    print(classification_report(test_y, predict_y))


def main():
    # 数据集路径
    filename_500 = "E:/video_traffic_datas/mix_data/Early_training_data_set.pcap.Mixed_rseed-22.pcap.flow.stat.500.csv"
    filename_2000 = "E:/video_traffic_datas/mix_data/Early_training_data_set.pcap.Mixed_rseed-22.pcap.flow.stat.2000.csv"
    filename_5000 = "E:/video_traffic_datas/mix_data/Early_training_data_set.pcap.Mixed_rseed-22.pcap.flow.stat.5000.csv"

    # 分别训练三级模型
    # main_stat_500(filename_500)
    # main_stat_2000(filename_2000)
    rf_clf_model = main_stat_5000(filename_5000)

    # 4 保存模型
    # joblib.dump(rf_clf_model, "./models/rf_clf_5000_1226.pkl")

    # 5 结果可视化
    # plot_roc(df_labels, predict_label)
    # plot_pr(df_labels, predict_label)
    # plot_importance_MDI(rf_clf_model, df_features.columns)
    # print_decision_tree_image(rf_clf_model.estimators_[0], 'decision_tree.png')


if __name__ == "__main__":
    main()
>>>>>>> 74d556a (tools)
