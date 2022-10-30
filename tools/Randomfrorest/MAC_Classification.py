<<<<<<< HEAD
import csv
import numpy as np
import pandas as pd
from collections import Counter
import time
import random

from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import graphviz

# start_time = time.time()
#-——————————————————————------------data_write_csv用于写入文件————————————————————————————--------------------------------————————————-----------————————————-----------————————————-----------
def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        for data in datas:
            writer.writerow(data)  # writerow()方法是一行一行写入.
    # print("保存文件成功，处理结束")


#------------------------------------------------数据处理--------------------------------------------------------------------
dfOrigin_normal = pd.read_csv("../../../0610_Thre50/202006101400.pcap.cut_type0_0-450_payload0.pcap.Ratio_128.RS_2222.BC_28.ST_50.MACP.csv", index_col=False)
#dfOrigin_normal = pd.read_csv("../../../0610_firstMixed/normal/202006101400.pcap.cut_type0_450--1_payload0.pcap.Ratio_128.RS_2222.BC_28.ST_100.MACP.csv", index_col=False)
df_normal = dfOrigin_normal.loc[dfOrigin_normal['ID']==972285968]  #40:972285968
df_normal1 = df_normal.drop(index=(df_normal.loc[(df_normal['F_Pck_Spd']==-1.00000)].index))
df_normal1['label'] = 0
dfCounters_normal = df_normal1[['F_Pck','B_Pck','F_Hash_IPT','B_Hash_IPT','F_Len_Avg','B_Len_Avg','F_Pck_Spd','B_Pck_Spd']]

dfOrigin_mix = pd.read_csv("../../../0610_Thre50/202006101400.pcap.cut_type0_450--1_payload0.pcap.Mixed_rseed-2222.pcap.cut_type0_1--1_payload0.pcap.Ratio_128.RS_2222.BC_28.ST_50.MACP.csv", index_col=False)
df_mix = dfOrigin_mix[dfOrigin_mix['ID']==972285968]
df_mix1 = df_mix.drop(index=(df_mix.loc[(df_mix['F_Pck_Spd']==-1.00000)].index))
df_mix1['label'] = 1
#df_mix1.loc[df_mix1['Time']<0.14588,'label'] = 0  #前半段背景流量混合的攻击流量，在混合开始时间之前的也是背景流量
dfCounters_mix = df_mix1[['F_Pck','B_Pck','F_Hash_IPT','B_Hash_IPT','F_Len_Avg','B_Len_Avg','F_Pck_Spd','B_Pck_Spd']]

X_data = dfCounters_normal.append(dfCounters_mix)
Y_data = df_normal1['label'].append(df_mix1['label'])
Xfeatures=np.array(X_data,dtype='float32') # 默认float64 减小内存
Ylabel = np.array(Y_data,dtype='float32')
'''
list_labeled = []  #list_labeled存放要写入的数据，包括自己标记的标签和模型预测的标签
column = list(df_normal1.columns)
column.append('predict_label')
list_labeled.append(column)
'''
#-----------------------------------------------特征获取----------------------------------------------------------------
'''
#将正常流量和混合流量的6个UDP特征存入同一个list[list]
list_Xfeatures = []
list_Yfeatures = []


#取正常流量的特征
for indexs in dfCounters_normal.index: #遍历所有行 将dataframe格式的转换为list[list ]
    list_Xfeatures.append(dfCounters_normal.loc[indexs].values[0:])#values[8:]从第8列至最后一列.
    list_labeled.append(list(df_normal1.loc[indexs].values[0:]))
    list_Yfeatures.append(0)


#取混合流量的特征
for indexs in dfCounters_mix.index: #遍历所有行 将dataframe格式的转换为list[list ]
    list_Xfeatures.append(dfCounters_mix.loc[indexs].values[0:])#values[8:]从第8列至最后一列.
    list_labeled.append(list(df_mix1.loc[indexs].values[0:]))
    list_Yfeatures.append(1)

Xfeatures=np.array(list_Xfeatures,dtype='float32') # 默认float64 减小内存
Ylabel = np.array(list_Yfeatures,dtype='float32')
'''
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xfeatures, Ylabel, test_size=0.3, random_state=22)

#--------------------------------------决策树---------------------------------------------------------------
#DecisionTreeClassifier参数：criterion衡量分支质量的指标，默认为gini，还可以设为entropy，但是计算速度较慢
#random_state和splitter分别用来控制随机模式和随即选项，这两个可以用来降低模型过拟合的可能性，但是对高维数据才有点用
#对决策树进行正确剪枝以防止过拟合：max_depth、min_samples、min_samples_split

#model = DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_leaf=11, splitter='best', random_state=25)
model = DecisionTreeClassifier(criterion='gini', max_depth=4,random_state=22)
#model = RandomForestClassifier(criterion="gini", max_depth = 20)


'''
#-----grid search begin-------
parameters = {'splitter':('best','random')
              ,'criterion':("gini","entropy")
              ,"max_depth":[*range(1,20)]
              ,'min_samples_leaf':[*range(1,50,5)]
}

model = DecisionTreeClassifier(random_state=30)
GS = GridSearchCV(model, parameters, cv=10) # cv交叉验证
GS.fit(Xtrain,Ytrain)
print(GS.best_params_)
print(GS.best_score_)
#-----grid search end---------

'''

model.fit(Xtrain, Ytrain)
score_train = model.score(Xtrain,Ytrain)
score_test = model.score(Xtest,Ytest)
print(score_train)  #0.923
print(score_test)   #0.8954

'''
feature_name = ['F_Pck','B_Pck','F_Hash_IPT','B_Hash_IPT','F_Len_Avg','B_Len_Avg','F_Pck_Spd','B_Pck_Spd']
cn = ['0','1']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (25,30), dpi=300)
plot_tree(model,feature_names = feature_name, class_names=cn,filled = True);
fig.savefig('DDoS_DTree_MAC.png')
'''

#--------------------------------------模型预测、csv写入---------------------------------------------------------------
'''
predict_label = model.predict(Xfeatures)
list_predictLabel = list(predict_label)

row = len(list_predictLabel)
for i in range(0,row):
    list_labeled[i+1].append(list_predictLabel[i])

data_write_csv("../../../MAC_data/labeled_feature_E4.csv", list_labeled)

'''








'''
model = DecisionTreeClassifier(criterion='gini', max_depth=19, random_state=30)
model = model.fit(Xtrain,Ytrain)
score_train = model.score(Xtrain,Ytrain)
print("score_train:"+str(score_train))
score = model.score(Xtest,Ytest)  #返回预测的准确度accuracy
print("score_test:"+str(score))

test = []
for i in range(25):
    model = DecisionTreeClassifier(max_depth=i+1,criterion='gini', random_state=30)
    model.fit(Xtrain,Ytrain)
    score = model.score(Xtest,Ytest)
    test.append(score)

plt.plot(range(1,26),test,color="red",label="max_depth")
plt.legend()
plt.show()
=======
import csv
import numpy as np
import pandas as pd
from collections import Counter
import time
import random

from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import graphviz

# start_time = time.time()
#-——————————————————————------------data_write_csv用于写入文件————————————————————————————--------------------------------————————————-----------————————————-----------————————————-----------
def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        for data in datas:
            writer.writerow(data)  # writerow()方法是一行一行写入.
    # print("保存文件成功，处理结束")


#------------------------------------------------数据处理--------------------------------------------------------------------
dfOrigin_normal = pd.read_csv("../../../0610_Thre50/202006101400.pcap.cut_type0_0-450_payload0.pcap.Ratio_128.RS_2222.BC_28.ST_50.MACP.csv", index_col=False)
#dfOrigin_normal = pd.read_csv("../../../0610_firstMixed/normal/202006101400.pcap.cut_type0_450--1_payload0.pcap.Ratio_128.RS_2222.BC_28.ST_100.MACP.csv", index_col=False)
df_normal = dfOrigin_normal.loc[dfOrigin_normal['ID']==972285968]  #40:972285968
df_normal1 = df_normal.drop(index=(df_normal.loc[(df_normal['F_Pck_Spd']==-1.00000)].index))
df_normal1['label'] = 0
dfCounters_normal = df_normal1[['F_Pck','B_Pck','F_Hash_IPT','B_Hash_IPT','F_Len_Avg','B_Len_Avg','F_Pck_Spd','B_Pck_Spd']]

dfOrigin_mix = pd.read_csv("../../../0610_Thre50/202006101400.pcap.cut_type0_450--1_payload0.pcap.Mixed_rseed-2222.pcap.cut_type0_1--1_payload0.pcap.Ratio_128.RS_2222.BC_28.ST_50.MACP.csv", index_col=False)
df_mix = dfOrigin_mix[dfOrigin_mix['ID']==972285968]
df_mix1 = df_mix.drop(index=(df_mix.loc[(df_mix['F_Pck_Spd']==-1.00000)].index))
df_mix1['label'] = 1
#df_mix1.loc[df_mix1['Time']<0.14588,'label'] = 0  #前半段背景流量混合的攻击流量，在混合开始时间之前的也是背景流量
dfCounters_mix = df_mix1[['F_Pck','B_Pck','F_Hash_IPT','B_Hash_IPT','F_Len_Avg','B_Len_Avg','F_Pck_Spd','B_Pck_Spd']]

X_data = dfCounters_normal.append(dfCounters_mix)
Y_data = df_normal1['label'].append(df_mix1['label'])
Xfeatures=np.array(X_data,dtype='float32') # 默认float64 减小内存
Ylabel = np.array(Y_data,dtype='float32')
'''
list_labeled = []  #list_labeled存放要写入的数据，包括自己标记的标签和模型预测的标签
column = list(df_normal1.columns)
column.append('predict_label')
list_labeled.append(column)
'''
#-----------------------------------------------特征获取----------------------------------------------------------------
'''
#将正常流量和混合流量的6个UDP特征存入同一个list[list]
list_Xfeatures = []
list_Yfeatures = []


#取正常流量的特征
for indexs in dfCounters_normal.index: #遍历所有行 将dataframe格式的转换为list[list ]
    list_Xfeatures.append(dfCounters_normal.loc[indexs].values[0:])#values[8:]从第8列至最后一列.
    list_labeled.append(list(df_normal1.loc[indexs].values[0:]))
    list_Yfeatures.append(0)


#取混合流量的特征
for indexs in dfCounters_mix.index: #遍历所有行 将dataframe格式的转换为list[list ]
    list_Xfeatures.append(dfCounters_mix.loc[indexs].values[0:])#values[8:]从第8列至最后一列.
    list_labeled.append(list(df_mix1.loc[indexs].values[0:]))
    list_Yfeatures.append(1)

Xfeatures=np.array(list_Xfeatures,dtype='float32') # 默认float64 减小内存
Ylabel = np.array(list_Yfeatures,dtype='float32')
'''
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xfeatures, Ylabel, test_size=0.3, random_state=22)

#--------------------------------------决策树---------------------------------------------------------------
#DecisionTreeClassifier参数：criterion衡量分支质量的指标，默认为gini，还可以设为entropy，但是计算速度较慢
#random_state和splitter分别用来控制随机模式和随即选项，这两个可以用来降低模型过拟合的可能性，但是对高维数据才有点用
#对决策树进行正确剪枝以防止过拟合：max_depth、min_samples、min_samples_split

#model = DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_leaf=11, splitter='best', random_state=25)
model = DecisionTreeClassifier(criterion='gini', max_depth=4,random_state=22)
#model = RandomForestClassifier(criterion="gini", max_depth = 20)


'''
#-----grid search begin-------
parameters = {'splitter':('best','random')
              ,'criterion':("gini","entropy")
              ,"max_depth":[*range(1,20)]
              ,'min_samples_leaf':[*range(1,50,5)]
}

model = DecisionTreeClassifier(random_state=30)
GS = GridSearchCV(model, parameters, cv=10) # cv交叉验证
GS.fit(Xtrain,Ytrain)
print(GS.best_params_)
print(GS.best_score_)
#-----grid search end---------

'''

model.fit(Xtrain, Ytrain)
score_train = model.score(Xtrain,Ytrain)
score_test = model.score(Xtest,Ytest)
print(score_train)  #0.923
print(score_test)   #0.8954

'''
feature_name = ['F_Pck','B_Pck','F_Hash_IPT','B_Hash_IPT','F_Len_Avg','B_Len_Avg','F_Pck_Spd','B_Pck_Spd']
cn = ['0','1']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (25,30), dpi=300)
plot_tree(model,feature_names = feature_name, class_names=cn,filled = True);
fig.savefig('DDoS_DTree_MAC.png')
'''

#--------------------------------------模型预测、csv写入---------------------------------------------------------------
'''
predict_label = model.predict(Xfeatures)
list_predictLabel = list(predict_label)

row = len(list_predictLabel)
for i in range(0,row):
    list_labeled[i+1].append(list_predictLabel[i])

data_write_csv("../../../MAC_data/labeled_feature_E4.csv", list_labeled)

'''








'''
model = DecisionTreeClassifier(criterion='gini', max_depth=19, random_state=30)
model = model.fit(Xtrain,Ytrain)
score_train = model.score(Xtrain,Ytrain)
print("score_train:"+str(score_train))
score = model.score(Xtest,Ytest)  #返回预测的准确度accuracy
print("score_test:"+str(score))

test = []
for i in range(25):
    model = DecisionTreeClassifier(max_depth=i+1,criterion='gini', random_state=30)
    model.fit(Xtrain,Ytrain)
    score = model.score(Xtest,Ytest)
    test.append(score)

plt.plot(range(1,26),test,color="red",label="max_depth")
plt.legend()
plt.show()
>>>>>>> 74d556a (tools)
'''