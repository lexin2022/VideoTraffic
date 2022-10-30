<<<<<<< HEAD
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import csv
import joblib


def plot_roc(labels, predict_prob):
    """
    函数说明：绘制ROC曲线
    Parameters:
         labels:测试标签列表
         predict_prob:预测标签列表
    """
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(labels, predict_prob)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)  # 计算AUC值
    print('AUC=' + str(roc_auc))
    plt.title('PC5-ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    # plt.savefig('figures/PC5.png') #将ROC图片进行保存
    plt.show()


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
    filename = "Early_training_data_set.pcap.Mixed_rseed-22.pcap.VS_IPP.stat.5000_TCP.csv"

    # 2 数据处理
    dfOrigin_data = pd.read_csv(path_fn + filename)
    df_features = dfOrigin_data[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl', 'o_spd_pck', 'i_spd_pck', 'o_spd_len', 'i_spd_len']]
    # print(df_features.head(5))
    df_labels = dfOrigin_data['label']
    X_features = np.array(df_features, dtype='float32')  # 默认float64 减小内存
    Y_labels = np.array(df_labels, dtype='float32')
    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # 3 记录被预测的原始数据
    list_features_labels = []  # 保存全部特征向量、自己标记的标签、模型预测的标签
    column = list(dfOrigin_data.columns)
    column.append('predict_label')
    column.append('FN|FP')
    column.append('correct')
    list_features_labels.append(column)
    for i_index in df_features.index:  # 遍历所有行 将dataframe格式的转换为list[list ]
        list_features_labels.append(list(dfOrigin_data.loc[i_index].values[0:]))

    # 4 训练模型并评估模型
    rf_clf_model = RandomForestClassifier(n_estimators=10000, random_state=0)
    rf_clf_model.fit(train_x, train_y)
    predict_label = rf_clf_model.predict(test_x)
    print(classification_report(test_y, predict_label))
    # plot_roc(test_y, predict_y)

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
    # data_write_csv("H:/Video service three-layer prediction(data)/Result4test/TCP_VS_IPP_stat_5000_FN_FP_20211109_v3.csv", list_features_labels)

    print(rf_clf_model.feature_importances_) # 输出特征重要性

    # 7 保存模型
    joblib.dump(rf_clf_model, "./models/rf_clf_5000_v3.pkl")


if __name__ == "__main__":
    main()
=======
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import csv
import joblib


def plot_roc(labels, predict_prob):
    """
    函数说明：绘制ROC曲线
    Parameters:
         labels:测试标签列表
         predict_prob:预测标签列表
    """
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(labels, predict_prob)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)  # 计算AUC值
    print('AUC=' + str(roc_auc))
    plt.title('PC5-ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    # plt.savefig('figures/PC5.png') #将ROC图片进行保存
    plt.show()


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
    filename = "Early_training_data_set.pcap.Mixed_rseed-22.pcap.VS_IPP.stat.5000_TCP.csv"

    # 2 数据处理
    dfOrigin_data = pd.read_csv(path_fn + filename)
    df_features = dfOrigin_data[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl', 'o_spd_pck', 'i_spd_pck', 'o_spd_len', 'i_spd_len']]
    # print(df_features.head(5))
    df_labels = dfOrigin_data['label']
    X_features = np.array(df_features, dtype='float32')  # 默认float64 减小内存
    Y_labels = np.array(df_labels, dtype='float32')
    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # 3 记录被预测的原始数据
    list_features_labels = []  # 保存全部特征向量、自己标记的标签、模型预测的标签
    column = list(dfOrigin_data.columns)
    column.append('predict_label')
    column.append('FN|FP')
    column.append('correct')
    list_features_labels.append(column)
    for i_index in df_features.index:  # 遍历所有行 将dataframe格式的转换为list[list ]
        list_features_labels.append(list(dfOrigin_data.loc[i_index].values[0:]))

    # 4 训练模型并评估模型
    rf_clf_model = RandomForestClassifier(n_estimators=10000, random_state=0)
    rf_clf_model.fit(train_x, train_y)
    predict_label = rf_clf_model.predict(test_x)
    print(classification_report(test_y, predict_label))
    # plot_roc(test_y, predict_y)

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
    # data_write_csv("H:/Video service three-layer prediction(data)/Result4test/TCP_VS_IPP_stat_5000_FN_FP_20211109_v3.csv", list_features_labels)

    print(rf_clf_model.feature_importances_) # 输出特征重要性

    # 7 保存模型
    joblib.dump(rf_clf_model, "./models/rf_clf_5000_v3.pkl")


if __name__ == "__main__":
    main()
>>>>>>> 74d556a (tools)
