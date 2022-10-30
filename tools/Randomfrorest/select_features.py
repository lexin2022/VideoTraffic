<<<<<<< HEAD
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler    # 标准化归一数据


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

    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title('P-R curve', fontsize=25)

    plt.xticks(rotation=45)

    plt.show()


# Plot feature importance
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


# Plot training deviance
def plot_training_deviance(clf, n_estimators, X_test, y_test):
    # compute test set deviance
    test_score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    train_score = clf.train_score_
    logging.info("len(train_score): %s" % len(train_score))
    logging.info(train_score)
    logging.info("len(test_score): %s" % len(test_score))
    logging.info(test_score)
    plt.plot(np.arange(n_estimators) + 1, train_score, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(n_estimators) + 1, test_score, 'r*', label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.show()


def main():
    # 1 读取文件
    path_fn = "E:/video_traffic_datas/mix_data/"
    filename = "capture_video_flows.pcap.flow.stat.2000.csv"

    # 2 划分训练集、测试集
    dfOrigin_data = pd.read_csv(path_fn + filename)
    df_features = dfOrigin_data.iloc[:, 7:55]
    df_features.drop(columns=['o_pck', 'i_pck', 'o_data_p', 'i_data_p', 'o_len', 'i_len', 'o_data_l', 'i_data_l',
                              'begin TM', 'end TM'], inplace=True)
    print(df_features.head())
    df_labels = dfOrigin_data['label']
    X_features = np.array(df_features, dtype='float32')  # 默认float64 减小内存
    Y_labels = np.array(df_labels, dtype='float32')
    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # 3 模型拟合
    feature_names = df_features.columns
    forest = RandomForestClassifier(n_estimators=200, random_state=0)
    forest.fit(train_x, train_y)

    # # 4 Feature importance based on feature permutation
    # start_time = time.time()
    # result = permutation_importance(forest, test_x, test_y, n_repeats=10, random_state=42, n_jobs=-1)
    # elapsed_time = time.time() - start_time
    # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    #
    # forest_importances = pd.Series(result.importances_mean, index=feature_names)
    #
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model")
    # ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    # plt.show()

    # 5 Feature importance based on mean decrease in impurity
    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances based on MDI", fontsize=20)
    ax.set_ylabel("Mean decrease in impurity", fontsize=15)

    ax.set_xticklabels(labels=feature_names, rotation=45, fontsize=15, family='SimHei')

    fig.tight_layout()
    plt.show()

    # 6打印分类结果评估报告
    # predict_label = forest.predict(test_x)
    # print(classification_report(test_y, predict_label))

    # print("Features sorted by their score:")
    # print(sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), feature_names), reverse=True))


if __name__ == "__main__":
    main()
=======
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler    # 标准化归一数据


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

    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title('P-R curve', fontsize=25)

    plt.xticks(rotation=45)

    plt.show()


# Plot feature importance
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


# Plot training deviance
def plot_training_deviance(clf, n_estimators, X_test, y_test):
    # compute test set deviance
    test_score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    train_score = clf.train_score_
    logging.info("len(train_score): %s" % len(train_score))
    logging.info(train_score)
    logging.info("len(test_score): %s" % len(test_score))
    logging.info(test_score)
    plt.plot(np.arange(n_estimators) + 1, train_score, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(n_estimators) + 1, test_score, 'r*', label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.show()


def main():
    # 1 读取文件
    path_fn = "E:/video_traffic_datas/mix_data/"
    filename = "capture_video_flows.pcap.flow.stat.2000.csv"

    # 2 划分训练集、测试集
    dfOrigin_data = pd.read_csv(path_fn + filename)
    df_features = dfOrigin_data.iloc[:, 7:55]
    df_features.drop(columns=['o_pck', 'i_pck', 'o_data_p', 'i_data_p', 'o_len', 'i_len', 'o_data_l', 'i_data_l',
                              'begin TM', 'end TM'], inplace=True)
    print(df_features.head())
    df_labels = dfOrigin_data['label']
    X_features = np.array(df_features, dtype='float32')  # 默认float64 减小内存
    Y_labels = np.array(df_labels, dtype='float32')
    train_x, test_x, train_y, test_y = train_test_split(X_features, Y_labels, test_size=0.25, random_state=1)

    # 3 模型拟合
    feature_names = df_features.columns
    forest = RandomForestClassifier(n_estimators=200, random_state=0)
    forest.fit(train_x, train_y)

    # # 4 Feature importance based on feature permutation
    # start_time = time.time()
    # result = permutation_importance(forest, test_x, test_y, n_repeats=10, random_state=42, n_jobs=-1)
    # elapsed_time = time.time() - start_time
    # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    #
    # forest_importances = pd.Series(result.importances_mean, index=feature_names)
    #
    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model")
    # ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    # plt.show()

    # 5 Feature importance based on mean decrease in impurity
    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances based on MDI", fontsize=20)
    ax.set_ylabel("Mean decrease in impurity", fontsize=15)

    ax.set_xticklabels(labels=feature_names, rotation=45, fontsize=15, family='SimHei')

    fig.tight_layout()
    plt.show()

    # 6打印分类结果评估报告
    # predict_label = forest.predict(test_x)
    # print(classification_report(test_y, predict_label))

    # print("Features sorted by their score:")
    # print(sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), feature_names), reverse=True))


if __name__ == "__main__":
    main()
>>>>>>> 74d556a (tools)
