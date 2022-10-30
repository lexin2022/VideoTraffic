<<<<<<< HEAD
import numpy as np
import pandas as pd
import csv
import joblib
from sklearn.preprocessing import StandardScaler


# # 1 预测：stat = 1000
# rf_clf_model_1000 = joblib.load("./models/rf_clf_1000_v3.pkl")
# df_stat_1000 = pd.read_csv("H:/Video service three-layer prediction(data)/stat_data_20211116/"
#                            "dump.CONN.pcap.VS_IPP.stat.1000_TCP.csv")
# df_features = df_stat_1000[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl']]
# X_features = np.array(df_features, dtype='float32')
#
# label_list = []
# # predict_label = (rf_clf_model_1000.predict_proba(X_features)[:, 1] >= 0.3).astype(int)
# predict_label = rf_clf_model_1000.predict(X_features)
# for index in predict_label:
#     label_list.append(index)
# df_stat_1000["label"] = label_list
#
# proba_list = []
# probability = rf_clf_model_1000.predict_proba(X_features)[:, 1]
# for proba in probability:
#     proba_list.append(proba)
# df_stat_1000["probability"] = proba_list
#
# df_stat_1000.to_csv("H:/Video service three-layer prediction(data)/Result4test/20211116/"
#                     "TCP_VS_predict_result.stat.1000_v2.csv", index=False, sep=',')

# # 2 预测：stat = 5000
# rf_clf_model_5000 = joblib.load("./models/rf_clf_5000_v3.pkl")
# df_stat_5000 = pd.read_csv("H:/Video service three-layer prediction(data)/stat_data_20211116/"
#                            "dump.CONN.pcap.VS_IPP.stat.5000_TCP.csv")
# df_features = df_stat_5000[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl', 'o_spd_pck', 'i_spd_pck', 'o_spd_len', 'i_spd_len']]
# X_features = np.array(df_features, dtype='float32')
# label_list = []
# # predict_label = (rf_clf_model_5000.predict_proba(X_features)[:, 1] >= 0.3).astype(int)
# predict_label = rf_clf_model_5000.predict(X_features)
#
# for index in predict_label:
#     label_list.append(index)
# df_stat_5000["label"] = label_list
#
# proba_list = []
# probability = rf_clf_model_5000.predict_proba(X_features)[:, 1]
# for proba in probability:
#     proba_list.append(proba)
# df_stat_5000["probability"] = proba_list
#
# df_stat_5000.to_csv("H:/Video service three-layer prediction(data)/Result4test/20211116/"
#                     "TCP_VS_predict_result.stat.5000_v2.csv", index=False, sep=',')

# # 3 预测：stat = 20000
# rf_clf_model_20000 = joblib.load("./models/rf_clf_20000_v3.pkl")
# df_stat_20000 = pd.read_csv("H:/Video service three-layer prediction(data)/stat_data_20211116/"
#                            "dump.CONN.pcap.VS_IPP.stat.20000_TCP.csv")
# df_features_part1 = df_stat_20000[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl', 'o_spd_pck', 'i_spd_pck', 'o_spd_len', 'i_spd_len']]
# df_features_part2 = df_stat_20000.iloc[0:, 23:53]
# df_features = pd.concat([df_features_part1, df_features_part2], axis=1)
# print(df_features.head(5))
# X_features = np.array(df_features, dtype='float32')
# label_list = []
# # predict_label = (rf_clf_model_20000.predict_proba(X_features)[:, 1] >= 0.3).astype(int)
# predict_label = rf_clf_model_20000.predict(X_features)
#
# for index in predict_label:
#     label_list.append(index)
# df_stat_20000["label"] = label_list
#
# proba_list = []
# probability = rf_clf_model_20000.predict_proba(X_features)[:, 1]
# for proba in probability:
#     proba_list.append(proba)
# df_stat_20000["probability"] = proba_list
#
# df_stat_20000.to_csv("H:/Video service three-layer prediction(data)/Result4test/20211116/"
#                     "TCP_VS_predict_result.stat.20000_v2.csv", index=False, sep=',')


rf_clf_model_5000 = joblib.load("./models/rf_clf_5000_1226.pkl")

dfOrigin_data = pd.read_csv("H:/video_traffic_datas/cernet3/cernt_per50/2021_11_28_12_00_lasts_400s.pcap.flow.stat.5000.csv")
df_features = dfOrigin_data[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl',
                             'o_spd_pck', 'i_spd_pck', 'o_spd_len', 'i_spd_len']]
df_features['per_o_(0)'] = dfOrigin_data[['per_o_(0)']]
df_features['per_o_(1-100)'] = dfOrigin_data[['per_o_(1-100)']]
df_features['per_o_(101-700)'] = dfOrigin_data.loc[:, 'per_o_(101-200)':'per_o_(601-700)'].sum(axis=1)
df_features['per_o_(701-1300)'] = dfOrigin_data.loc[:, 'per_o_(701-800)':'per_o_(1201-1300)'].sum(axis=1)
df_features['per_o_(>1300)'] = dfOrigin_data['per_o_(>1300)']
df_features['per_i_(0)'] = dfOrigin_data[['per_i_(0)']]
df_features['per_i_(1-100)'] = dfOrigin_data[['per_i_(1-100)']]
df_features['per_i_(101-700)'] = dfOrigin_data.loc[:, 'per_i_(101-200)':'per_i_(601-700)'].sum(axis=1)
df_features['per_i_(701-1300)'] = dfOrigin_data.loc[:, 'per_i_(701-800)':'per_i_(1201-1300)'].sum(axis=1)
df_features['per_i_(>1300)'] = dfOrigin_data['per_i_(>1300)']

X_features = np.array(df_features, dtype='float32')
standard = StandardScaler()
standard.fit(X_features)
X_features = standard.transform(X_features)
label_list = []
predict_label = rf_clf_model_5000.predict(X_features)

for index in predict_label:
    label_list.append(index)
dfOrigin_data["label"] = label_list

proba_list = []
probability = rf_clf_model_5000.predict_proba(X_features)[:, 1]
for proba in probability:
    proba_list.append(proba)
dfOrigin_data["probability"] = proba_list

dfOrigin_data.to_csv("H:/cernet3/cernt_per50/"
=======
import numpy as np
import pandas as pd
import csv
import joblib
from sklearn.preprocessing import StandardScaler


# # 1 预测：stat = 1000
# rf_clf_model_1000 = joblib.load("./models/rf_clf_1000_v3.pkl")
# df_stat_1000 = pd.read_csv("H:/Video service three-layer prediction(data)/stat_data_20211116/"
#                            "dump.CONN.pcap.VS_IPP.stat.1000_TCP.csv")
# df_features = df_stat_1000[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl']]
# X_features = np.array(df_features, dtype='float32')
#
# label_list = []
# # predict_label = (rf_clf_model_1000.predict_proba(X_features)[:, 1] >= 0.3).astype(int)
# predict_label = rf_clf_model_1000.predict(X_features)
# for index in predict_label:
#     label_list.append(index)
# df_stat_1000["label"] = label_list
#
# proba_list = []
# probability = rf_clf_model_1000.predict_proba(X_features)[:, 1]
# for proba in probability:
#     proba_list.append(proba)
# df_stat_1000["probability"] = proba_list
#
# df_stat_1000.to_csv("H:/Video service three-layer prediction(data)/Result4test/20211116/"
#                     "TCP_VS_predict_result.stat.1000_v2.csv", index=False, sep=',')

# # 2 预测：stat = 5000
# rf_clf_model_5000 = joblib.load("./models/rf_clf_5000_v3.pkl")
# df_stat_5000 = pd.read_csv("H:/Video service three-layer prediction(data)/stat_data_20211116/"
#                            "dump.CONN.pcap.VS_IPP.stat.5000_TCP.csv")
# df_features = df_stat_5000[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl', 'o_spd_pck', 'i_spd_pck', 'o_spd_len', 'i_spd_len']]
# X_features = np.array(df_features, dtype='float32')
# label_list = []
# # predict_label = (rf_clf_model_5000.predict_proba(X_features)[:, 1] >= 0.3).astype(int)
# predict_label = rf_clf_model_5000.predict(X_features)
#
# for index in predict_label:
#     label_list.append(index)
# df_stat_5000["label"] = label_list
#
# proba_list = []
# probability = rf_clf_model_5000.predict_proba(X_features)[:, 1]
# for proba in probability:
#     proba_list.append(proba)
# df_stat_5000["probability"] = proba_list
#
# df_stat_5000.to_csv("H:/Video service three-layer prediction(data)/Result4test/20211116/"
#                     "TCP_VS_predict_result.stat.5000_v2.csv", index=False, sep=',')

# # 3 预测：stat = 20000
# rf_clf_model_20000 = joblib.load("./models/rf_clf_20000_v3.pkl")
# df_stat_20000 = pd.read_csv("H:/Video service three-layer prediction(data)/stat_data_20211116/"
#                            "dump.CONN.pcap.VS_IPP.stat.20000_TCP.csv")
# df_features_part1 = df_stat_20000[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl', 'o_spd_pck', 'i_spd_pck', 'o_spd_len', 'i_spd_len']]
# df_features_part2 = df_stat_20000.iloc[0:, 23:53]
# df_features = pd.concat([df_features_part1, df_features_part2], axis=1)
# print(df_features.head(5))
# X_features = np.array(df_features, dtype='float32')
# label_list = []
# # predict_label = (rf_clf_model_20000.predict_proba(X_features)[:, 1] >= 0.3).astype(int)
# predict_label = rf_clf_model_20000.predict(X_features)
#
# for index in predict_label:
#     label_list.append(index)
# df_stat_20000["label"] = label_list
#
# proba_list = []
# probability = rf_clf_model_20000.predict_proba(X_features)[:, 1]
# for proba in probability:
#     proba_list.append(proba)
# df_stat_20000["probability"] = proba_list
#
# df_stat_20000.to_csv("H:/Video service three-layer prediction(data)/Result4test/20211116/"
#                     "TCP_VS_predict_result.stat.20000_v2.csv", index=False, sep=',')


rf_clf_model_5000 = joblib.load("./models/rf_clf_5000_1226.pkl")

dfOrigin_data = pd.read_csv("H:/video_traffic_datas/cernet3/cernt_per50/2021_11_28_12_00_lasts_400s.pcap.flow.stat.5000.csv")
df_features = dfOrigin_data[['r_o_pck', 'r_o_dp', 'r_o_len', 'r_o_dl',
                             'o_spd_pck', 'i_spd_pck', 'o_spd_len', 'i_spd_len']]
df_features['per_o_(0)'] = dfOrigin_data[['per_o_(0)']]
df_features['per_o_(1-100)'] = dfOrigin_data[['per_o_(1-100)']]
df_features['per_o_(101-700)'] = dfOrigin_data.loc[:, 'per_o_(101-200)':'per_o_(601-700)'].sum(axis=1)
df_features['per_o_(701-1300)'] = dfOrigin_data.loc[:, 'per_o_(701-800)':'per_o_(1201-1300)'].sum(axis=1)
df_features['per_o_(>1300)'] = dfOrigin_data['per_o_(>1300)']
df_features['per_i_(0)'] = dfOrigin_data[['per_i_(0)']]
df_features['per_i_(1-100)'] = dfOrigin_data[['per_i_(1-100)']]
df_features['per_i_(101-700)'] = dfOrigin_data.loc[:, 'per_i_(101-200)':'per_i_(601-700)'].sum(axis=1)
df_features['per_i_(701-1300)'] = dfOrigin_data.loc[:, 'per_i_(701-800)':'per_i_(1201-1300)'].sum(axis=1)
df_features['per_i_(>1300)'] = dfOrigin_data['per_i_(>1300)']

X_features = np.array(df_features, dtype='float32')
standard = StandardScaler()
standard.fit(X_features)
X_features = standard.transform(X_features)
label_list = []
predict_label = rf_clf_model_5000.predict(X_features)

for index in predict_label:
    label_list.append(index)
dfOrigin_data["label"] = label_list

proba_list = []
probability = rf_clf_model_5000.predict_proba(X_features)[:, 1]
for proba in probability:
    proba_list.append(proba)
dfOrigin_data["probability"] = proba_list

dfOrigin_data.to_csv("H:/cernet3/cernt_per50/"
>>>>>>> 74d556a (tools)
                    "TCP_VS_predict_result.stat.5000_1226.csv", index=False, sep=',')