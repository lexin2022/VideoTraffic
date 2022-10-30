<<<<<<< HEAD
import numpy as np
import pandas as pd
import csv
import joblib
from sklearn.preprocessing import StandardScaler


def sample_data_predict(filename, model):
    dataframe = pd.read_csv(filename + '.csv')
    df_features = dataframe[['per_i_(0)', 'r_o_dp', 'r_o_dl', 'per_o_(>1300)', 'r_o_len', 'per_i_(1-100)',
                             'o_spd_len', 'r_o_pck']]
    X_features = np.array(df_features)
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    predict_label = model.predict(X_features)
    label_list = np.array(predict_label, dtype=int)
    probability = model.predict_proba(X_features)[:, 1]
    proba_list = np.array(probability)

    dataframe['predict_label'] = label_list
    dataframe['predict_proba'] = proba_list

    dataframe.to_excel(filename + '.predict' + '.xlsx', index=None)

    labeled = dataframe['label']
    cnt_video, cnt_recall_video = 0, 0
    for i in range(len(labeled)):
        if labeled[i]:
            cnt_video += 1
            if labeled[i] == label_list[i]:
                cnt_recall_video += 1
    print('model predict number of total video flow is {0}'.format(sum(label_list)))
    print('cnt_video = {0}, cnt_recall_video = {1}'.format(cnt_video, cnt_recall_video))
    print('recall = {0}'.format(cnt_recall_video/cnt_video))


model = joblib.load("./models/model.f1_per9886.pkl")

fn_ratio_1 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_1.thre_2000.s_2222.mix_videoflow"
fn_ratio_8 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_8.thre_2000.s_2222.mix_videoflow"
fn_ratio_16 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_16.thre_1000.s_2222.mix_videoflow"
fn_ratio_32 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_32.thre_500.s_2222.mix_videoflow"
fn_ratio_64 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_64.thre_250.s_2222. mix_videoflow"
fn_ratio_128 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_128.thre_100.s_2222.mix_videoflow"

=======
import numpy as np
import pandas as pd
import csv
import joblib
from sklearn.preprocessing import StandardScaler


def sample_data_predict(filename, model):
    dataframe = pd.read_csv(filename + '.csv')
    df_features = dataframe[['per_i_(0)', 'r_o_dp', 'r_o_dl', 'per_o_(>1300)', 'r_o_len', 'per_i_(1-100)',
                             'o_spd_len', 'r_o_pck']]
    X_features = np.array(df_features)
    standard = StandardScaler()
    standard.fit(X_features)
    X_features = standard.transform(X_features)

    predict_label = model.predict(X_features)
    label_list = np.array(predict_label, dtype=int)
    probability = model.predict_proba(X_features)[:, 1]
    proba_list = np.array(probability)

    dataframe['predict_label'] = label_list
    dataframe['predict_proba'] = proba_list

    dataframe.to_excel(filename + '.predict' + '.xlsx', index=None)

    labeled = dataframe['label']
    cnt_video, cnt_recall_video = 0, 0
    for i in range(len(labeled)):
        if labeled[i]:
            cnt_video += 1
            if labeled[i] == label_list[i]:
                cnt_recall_video += 1
    print('model predict number of total video flow is {0}'.format(sum(label_list)))
    print('cnt_video = {0}, cnt_recall_video = {1}'.format(cnt_video, cnt_recall_video))
    print('recall = {0}'.format(cnt_recall_video/cnt_video))


model = joblib.load("./models/model.f1_per9886.pkl")

fn_ratio_1 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_1.thre_2000.s_2222.mix_videoflow"
fn_ratio_8 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_8.thre_2000.s_2222.mix_videoflow"
fn_ratio_16 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_16.thre_1000.s_2222.mix_videoflow"
fn_ratio_32 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_32.thre_500.s_2222.mix_videoflow"
fn_ratio_64 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_64.thre_250.s_2222. mix_videoflow"
fn_ratio_128 = "E:/video_traffic_datas/cernet3/cernet-mix_video/flow.stat.r_128.thre_100.s_2222.mix_videoflow"

>>>>>>> 74d556a (tools)
sample_data_predict(fn_ratio_128, model)