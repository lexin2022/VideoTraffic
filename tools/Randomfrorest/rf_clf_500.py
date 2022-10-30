<<<<<<< HEAD
import pandas as pd
import numpy as np
import csv
import joblib
import matplotlib.pyplot as plt
import time
import os

from lazypredict.Supervised import LazyClassifier  #sklearn 批量模型拟合
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler    #归一化函数
from sklearn.inspection import permutation_importance   #特征重要性
from sklearn.tree import export_text, export_graphviz
import pydotplus    #画图工具，用于绘制决策树图像

# pd.set_option('display.width', 1000)  # 加了这一行那表格的一行就不
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


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

def plot_importance_MDI(model, feature_names):
    # Feature importance based on mean decrease in impurity
    start_time = time.time()
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


def print_decision_tree_image(tree, out_path):
    # 将决策树绘制成图到你当前的路径
    # 加入Graphviz的环境路径
    os.environ["PATH"] += os.pathsep + "E:/Graphviz/bin"

    # 绘图并导出
    dot_data = export_graphviz(tree, out_file=None, feature_names = df_features.columns,
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


# -——————————————————————------------data_write_csv用于写入文件————————————————————————————--------------------------------————————————-----------————————————-----------————————————-----------
def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name, 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        for data in datas:
            writer.writerow(data)  # writerow()方法是一行一行写入.
    # print("保存文件成功，处理结束")


def main():
    # 1 读取文件
    path_fn = "H:/Video service three-layer prediction(data)/stat_data_20211115/"
    filename = "Early_training_data_set.pcap.Mixed_rseed-22.pcap.VS_IPP.stat.1000_TCP.csv"

    # 2 划分训练集、测试集
    dfOrigin_data = pd.read_csv(path_fn + filename)
    df_features = dfOrigin_data[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl']]
    df_labels = dfOrigin_data['label']
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')

    # 2.1 特征均值方差归一化
    # standard = StandardScaler()
    # standard.fit(X_features)
    # X_features = standard.transform(X_features)
    # print(X_features)
    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # # 3 记录被预测的原始数据
    # list_features_labels = []  # 保存全部特征向量、自己标记的标签、模型预测的标签
    # column = list(dfOrigin_data.columns)
    # column.append('predict_label')
    # column.append('FN|FP')
    # column.append('correct')
    # list_features_labels.append(column)
    # for i_index in df_features.index:  # 遍历所有行 将dataframe格式的转换为list[list ]
    #     list_features_labels.append(list(dfOrigin_data.loc[i_index].values[0:]))

    # 4 训练模型并评估模型
    # rf_clf_model = RandomForestClassifier(n_estimators=200, random_state=0)
    # rf_clf_model.fit(train_x, train_y)
    # predict_label = rf_clf_model.predict(test_x)
    # print(classification_report(test_y, predict_label))
    clf_model = LazyClassifier(predictions=True)
    models, predictions = clf_model.fit(train_x, test_x, train_y, test_y)
    print(models)

    # # 5 模型预测，并将结果写入csv文件
    # list_predict_label = list(predict_label)
    # row = len(list_predict_label)
    # for i in range(0, row):
    #     result = 'T'
    #     origin_label, cur_label = Y_labels[i], list_predict_label[i]
    #     list_features_labels[i + 1].append(cur_label)
    #     if origin_label != cur_label:
    #         if cur_label == 1:
    #             result = 'Fp'
    #         if cur_label == 0:
    #             result = 'FN'
    #     list_features_labels[i + 1].append(result)
    # data_write_csv("H:/Video service three-layer prediction(data)/Result4test/20211111/TCP_VS_IPP_stat_1000_FN_FP_20211111_v1.csv", list_features_labels)

    # # 6 调试模型参数
    # n_estimator_params = range(1000, 10000, 1000)
    # for n_estimator in n_estimator_params:
    #     rf_clf_model = RandomForestClassifier(n_estimators=n_estimator, n_jobs=-1, random_state=0, verbose=True)
    #     rf_clf_model.fit(train_x, train_y)
    #     predict_label = rf_clf_model.predict(X_features)
    #     print(classification_report(Y_labels, predict_label))

    # # 7 保存模型
    # joblib.dump(rf_clf_model, "./models/rf_clf_1000_v3.pkl")

    # 8 结果可视化
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

from lazypredict.Supervised import LazyClassifier  #sklearn 批量模型拟合
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler    #归一化函数
from sklearn.inspection import permutation_importance   #特征重要性
from sklearn.tree import export_text, export_graphviz
import pydotplus    #画图工具，用于绘制决策树图像

# pd.set_option('display.width', 1000)  # 加了这一行那表格的一行就不
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


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

def plot_importance_MDI(model, feature_names):
    # Feature importance based on mean decrease in impurity
    start_time = time.time()
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


def print_decision_tree_image(tree, out_path):
    # 将决策树绘制成图到你当前的路径
    # 加入Graphviz的环境路径
    os.environ["PATH"] += os.pathsep + "E:/Graphviz/bin"

    # 绘图并导出
    dot_data = export_graphviz(tree, out_file=None, feature_names = df_features.columns,
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


# -——————————————————————------------data_write_csv用于写入文件————————————————————————————--------------------------------————————————-----------————————————-----------————————————-----------
def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name, 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        for data in datas:
            writer.writerow(data)  # writerow()方法是一行一行写入.
    # print("保存文件成功，处理结束")


def main():
    # 1 读取文件
    path_fn = "H:/Video service three-layer prediction(data)/stat_data_20211115/"
    filename = "Early_training_data_set.pcap.Mixed_rseed-22.pcap.VS_IPP.stat.1000_TCP.csv"

    # 2 划分训练集、测试集
    dfOrigin_data = pd.read_csv(path_fn + filename)
    df_features = dfOrigin_data[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl']]
    df_labels = dfOrigin_data['label']
    X_features = np.array(df_features, dtype='float32')  # 创建一个数组
    Y_labels = np.array(df_labels, dtype='float32')

    # 2.1 特征均值方差归一化
    # standard = StandardScaler()
    # standard.fit(X_features)
    # X_features = standard.transform(X_features)
    # print(X_features)
    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # # 3 记录被预测的原始数据
    # list_features_labels = []  # 保存全部特征向量、自己标记的标签、模型预测的标签
    # column = list(dfOrigin_data.columns)
    # column.append('predict_label')
    # column.append('FN|FP')
    # column.append('correct')
    # list_features_labels.append(column)
    # for i_index in df_features.index:  # 遍历所有行 将dataframe格式的转换为list[list ]
    #     list_features_labels.append(list(dfOrigin_data.loc[i_index].values[0:]))

    # 4 训练模型并评估模型
    # rf_clf_model = RandomForestClassifier(n_estimators=200, random_state=0)
    # rf_clf_model.fit(train_x, train_y)
    # predict_label = rf_clf_model.predict(test_x)
    # print(classification_report(test_y, predict_label))
    clf_model = LazyClassifier(predictions=True)
    models, predictions = clf_model.fit(train_x, test_x, train_y, test_y)
    print(models)

    # # 5 模型预测，并将结果写入csv文件
    # list_predict_label = list(predict_label)
    # row = len(list_predict_label)
    # for i in range(0, row):
    #     result = 'T'
    #     origin_label, cur_label = Y_labels[i], list_predict_label[i]
    #     list_features_labels[i + 1].append(cur_label)
    #     if origin_label != cur_label:
    #         if cur_label == 1:
    #             result = 'Fp'
    #         if cur_label == 0:
    #             result = 'FN'
    #     list_features_labels[i + 1].append(result)
    # data_write_csv("H:/Video service three-layer prediction(data)/Result4test/20211111/TCP_VS_IPP_stat_1000_FN_FP_20211111_v1.csv", list_features_labels)

    # # 6 调试模型参数
    # n_estimator_params = range(1000, 10000, 1000)
    # for n_estimator in n_estimator_params:
    #     rf_clf_model = RandomForestClassifier(n_estimators=n_estimator, n_jobs=-1, random_state=0, verbose=True)
    #     rf_clf_model.fit(train_x, train_y)
    #     predict_label = rf_clf_model.predict(X_features)
    #     print(classification_report(Y_labels, predict_label))

    # # 7 保存模型
    # joblib.dump(rf_clf_model, "./models/rf_clf_1000_v3.pkl")

    # 8 结果可视化
    # plot_roc(df_labels, predict_label)
    # plot_pr(df_labels, predict_label)
    # plot_importance_MDI(rf_clf_model, df_features.columns)
    # print_decision_tree_image(rf_clf_model.estimators_[0], 'decision_tree.png')





if __name__ == "__main__":
    main()
>>>>>>> 74d556a (tools)
