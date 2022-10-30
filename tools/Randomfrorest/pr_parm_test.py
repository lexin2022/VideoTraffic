<<<<<<< HEAD
from sklearn.metrics import precision_recall_curve

y_true = [0, 0, 1, 1, 1]
y_score = [0.1, 0.4, 0.35, 0.8, 0.9]

precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
print(precisions)
print(recalls)
print(thresholds)

# F1 = 2 * precisions * recalls / (precisions + recalls)

max_F1 = 0
mask_index = 0
for index in range(len(recalls)):
    cur_F1 = 2 * precisions[index] * recalls[index] / (precisions[index] + recalls[index])
    if recalls[index] == 1 and cur_F1 > max_F1:
        max_F1 = cur_F1
        mask_index = index
=======
from sklearn.metrics import precision_recall_curve

y_true = [0, 0, 1, 1, 1]
y_score = [0.1, 0.4, 0.35, 0.8, 0.9]

precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
print(precisions)
print(recalls)
print(thresholds)

# F1 = 2 * precisions * recalls / (precisions + recalls)

max_F1 = 0
mask_index = 0
for index in range(len(recalls)):
    cur_F1 = 2 * precisions[index] * recalls[index] / (precisions[index] + recalls[index])
    if recalls[index] == 1 and cur_F1 > max_F1:
        max_F1 = cur_F1
        mask_index = index
>>>>>>> 74d556a (tools)
print("best threshold = {0}".format(thresholds[mask_index]))