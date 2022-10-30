<<<<<<< HEAD
import numpy as np
import pandas as pd
import csv
import joblib
from sklearn.preprocessing import StandardScaler
import time

quick_model = joblib.load("./models/quick_2000_packets.pkl")

start = time.time()
dataframe = pd.read_csv("E:/video_traffic_datas/cernet3/2021_11_28_12_00_lasts_400s.pcap.flow.stat.500.csv")
# dataframe = pd.read_csv("E:/Silhouette(paper)/silhouette-python/quic-_r9GPgvN8As-1080p.pcap.flow.stat.2000.csv")
df_features = dataframe[['per_i_(0)', 'r_o_dp', 'r_o_dl', 'per_o_(>1300)', 'r_o_len', 'per_i_(1-100)',
                         'o_spd_len', 'r_o_pck']]
X_features = np.array(df_features)
standard = StandardScaler()
standard.fit(X_features)
X_features = standard.transform(X_features)
end = time.time()
print(end - start)

start = time.time()
predict_label = quick_model.predict(X_features)
end = time.time()
print(end - start)

# label_list = np.array(predict_label, dtype=int)
# probability = quick_model.predict_proba(X_features)[:, 1]
# proba_list = np.array(probability)
#
# dataframe['predict_label'] = label_list
# dataframe['predict_proba'] = proba_list
#
# dataframe.to_csv("E:/Silhouette(paper)/silhouette-python/quic-_r9GPgvN8As-1080p.pcap.flow.stat.2000._predict.csv", index=None, sep=',')
#
# labeled = dataframe['label']
# cnt_video, cnt_recall_video = 0, 0
# for i in range(len(labeled)):
#     if labeled[i]:
#         cnt_video += 1
#         if labeled[i] == label_list[i]:
#             cnt_recall_video += 1
# print('model predict number of total video flow is {0}'.format(sum(label_list)))
# print('cnt_video = {0}, cnt_recall_video = {1}'.format(cnt_video, cnt_recall_video))
# print('recall = {0}'.format(cnt_recall_video/cnt_video))
=======
import numpy as np
import pandas as pd
import csv
import joblib
from sklearn.preprocessing import StandardScaler
import time

quick_model = joblib.load("./models/quick_2000_packets.pkl")

start = time.time()
dataframe = pd.read_csv("E:/video_traffic_datas/cernet3/2021_11_28_12_00_lasts_400s.pcap.flow.stat.500.csv")
# dataframe = pd.read_csv("E:/Silhouette(paper)/silhouette-python/quic-_r9GPgvN8As-1080p.pcap.flow.stat.2000.csv")
df_features = dataframe[['per_i_(0)', 'r_o_dp', 'r_o_dl', 'per_o_(>1300)', 'r_o_len', 'per_i_(1-100)',
                         'o_spd_len', 'r_o_pck']]
X_features = np.array(df_features)
standard = StandardScaler()
standard.fit(X_features)
X_features = standard.transform(X_features)
end = time.time()
print(end - start)

start = time.time()
predict_label = quick_model.predict(X_features)
end = time.time()
print(end - start)

# label_list = np.array(predict_label, dtype=int)
# probability = quick_model.predict_proba(X_features)[:, 1]
# proba_list = np.array(probability)
#
# dataframe['predict_label'] = label_list
# dataframe['predict_proba'] = proba_list
#
# dataframe.to_csv("E:/Silhouette(paper)/silhouette-python/quic-_r9GPgvN8As-1080p.pcap.flow.stat.2000._predict.csv", index=None, sep=',')
#
# labeled = dataframe['label']
# cnt_video, cnt_recall_video = 0, 0
# for i in range(len(labeled)):
#     if labeled[i]:
#         cnt_video += 1
#         if labeled[i] == label_list[i]:
#             cnt_recall_video += 1
# print('model predict number of total video flow is {0}'.format(sum(label_list)))
# print('cnt_video = {0}, cnt_recall_video = {1}'.format(cnt_video, cnt_recall_video))
# print('recall = {0}'.format(cnt_recall_video/cnt_video))
>>>>>>> 74d556a (tools)
