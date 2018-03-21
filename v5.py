# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:34:50 2018

@author: shivu.soman
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:33:35 2018

@author: shivu.soman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import seaborn as sns

dataset = pd.read_csv('new_processed_data_acc_ori_gyr_jayanth.csv')
df = dataset.dropna()


dataset2 = pd.read_csv('new_processed_data_acc_ori_gyr_our.csv')
df2 = dataset2.dropna()

dataset3 = pd.read_csv('new_processed_data_acc_ori_gyr_shweta.csv')
df3 = dataset3.dropna()

dataset4 = pd.read_csv('new_processed_data_acc_ori_gyr_jeet.csv')
df4 = dataset4.dropna()

dataset5 = pd.read_csv('new_processed_data_acc_ori_gyr_jeet2.csv')
df5 = dataset5.dropna()

X_1 = df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38]].values
y_1 = df.iloc[:,-1].values

#X_1.head()

X_2 = df2.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38]]
y_2 = df2.iloc[:,-1].values

X_3 = df3.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38]].values
y_3 = df3.iloc[:,-1].values

X_4 = df4.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38]].values
y_4 = df4.iloc[:,-1].values


X_5 = df5.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38]].values
y_5 = df5.iloc[:,-1].values

y_train2 = []
X_train2 = []
for i in range(len(y_2)):
    if y_2[i] == 'null':
        continue
    else:
        y_train2.append(y_2[i])
        X_train2.append(X_2[i])

y_train3 = []
X_train3 = []
for i in range(len(y_3)):
    if y_3[i] == 'null':
        continue
    else:
        y_train3.append(y_3[i])
        X_train3.append(X_3[i])
        
y_train4 = []
X_train4 = []
for i in range(len(y_4)):
    if y_4[i] == 'null':
        continue
    else:
        y_train4.append(y_4[i])
        X_train4.append(X_4[i])


y_test = []
X_test = []
for i in range(len(y_1)):
    if y_1[i] == 'null':
        continue
    else:
        y_test.append(y_1[i])
        X_test.append(X_1[i])
        
y_train5 = []
X_train5 = []
for i in range(len(y_5)):
    if y_5[i] == 'null':
        continue
    else:
        y_train5.append(y_5[i])
        X_train5.append(X_5[i])


        
X_train_all = X_test + X_train3 + X_train4 + X_train5
y_train_all = y_test + y_train3 + y_train4 + y_train5



feature_names = ['acc_xmean', 'acc_ymean', 'acc_zmean', 'acc_xstd', 'acc_ystd', 'acc_zstd', 'acc_xmax', 'acc_ymax', 'acc_zmax', 'acc_xmin', 'acc_ymin', 'acc_zmin', 'gyr_xmean', 'gyr_ymean', 'gyr_zmean', 'gyr_xstd', 'gyr_ystd', 'gyr_zstd', 'gyr_xmax', 'gyr_ymax', 'gyr_zmax', 'gyr_xmin', 'gyr_ymin', 'gyr_zmin', 'ori_xmean', 'ori_ymean', 'ori_zmean', 'ori_xstd', 'ori_ystd', 'ori_zstd', 'ori_xmax','ori_ymax', 'ori_zmax', 'ori_xmin', 'ori_ymin', 'ori_zmin']
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
kbest = SelectKBest(f_classif, k=15)
X_f = kbest.fit_transform(X_train_all, y_train_all)
fclassif = zip(kbest.scores_,feature_names)
print(sorted(fclassif,reverse=True))



X_2.head()
X_f_test = X_2.iloc[:,[3,4,5,9,10,11,15,16,17,18,19,20,21,22,23]].values
X_f_test2 = []
for i in range(len(y_2)):
    if y_2[i] == 'null':
        continue
    else:
        X_f_test2.append(X_f_test[i])

kbest = SelectKBest(mutual_info_classif, k=15)
X_mi = kbest.fit_transform(X_train_all, y_train_all)
miclassif = zip(kbest.scores_,feature_names)
print(sorted(miclassif,reverse=True))

X_mi_test = X_2.iloc[:,[0,3,4,5,6,9,15,16,17,20,23,26,27,32,35]].values
X_mi_test2 = []
for i in range(len(y_2)):
    if y_2[i] == 'null':
        continue
    else:
        X_mi_test2.append(X_mi_test[i])

clf1 = RandomForestClassifier(n_estimators = 100,random_state=0)
clf1.fit(X_train_all, y_train_all)
preds = clf1.predict(X_train2)
#print(preds)
print("Acc :",accuracy_score(y_train2,preds))
print("F1 score : ", f1_score(y_train2,preds,average='macro'))
print(classification_report(y_train2,preds))
LABELS = ['Laying Down', 'Sitting', 'Standing', 'Walking']
confusion_mat = confusion_matrix(y_train2,preds)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();

clf1.fit(X_f, y_train_all)
preds = clf1.predict(X_f_test2)
#print(preds)
print("Acc :",accuracy_score(y_train2,preds))
print("F1 score : ", f1_score(y_train2,preds,average='macro'))
print(classification_report(y_train2,preds))
LABELS = ['Laying Down', 'Sitting', 'Standing', 'Walking']
confusion_mat = confusion_matrix(y_train2,preds)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();

clf1.fit(X_mi, y_train_all)
preds = clf1.predict(X_mi_test2)
#print(preds)
print("Acc :",accuracy_score(y_train2,preds))
print("F1 score : ", f1_score(y_train2,preds,average='macro'))
print(classification_report(y_train2,preds))
LABELS = ['Laying Down', 'Sitting', 'Standing', 'Walking']
confusion_mat = confusion_matrix(y_train2,preds)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();

new = zip(clf1.feature_importances_,feature_names)
print(sorted(new,reverse=True))
print(np.argsort(clf1.feature_importances_))

clf2 = LogisticRegression(random_state=42)
clf2.fit(X_train_all, y_train_all)
preds = clf2.predict(X_train2)
#print(preds)
print("Acc :",accuracy_score(y_train2,preds))
print("F1 score : ", f1_score(y_train2,preds,average='macro'))
print(classification_report(y_train2,preds))
LABELS = ['Laying Down', 'Sitting', 'Standing', 'Walking']
confusion_mat = confusion_matrix(y_train2,preds)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();

clf2.fit(X_f, y_train_all)
preds = clf2.predict(X_f_test2)
#print(preds)
print("Acc :",accuracy_score(y_train2,preds))
print("F1 score : ", f1_score(y_train2,preds,average='macro'))
print(classification_report(y_train2,preds))
LABELS = ['Laying Down', 'Sitting', 'Standing', 'Walking']
confusion_mat = confusion_matrix(y_train2,preds)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();

clf2.fit(X_mi, y_train_all)
preds = clf2.predict(X_mi_test2)
#print(preds)
print("Acc :",accuracy_score(y_train2,preds))
print("F1 score : ", f1_score(y_train2,preds,average='macro'))
print(classification_report(y_train2,preds))
LABELS = ['Laying Down', 'Sitting', 'Standing', 'Walking']
confusion_mat = confusion_matrix(y_train2,preds)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();

clf3 = DecisionTreeClassifier(max_depth = 18,random_state=0)
clf3.fit(X_train_all, y_train_all)
preds = clf3.predict(X_train2)
#print(preds)
print("Acc :",accuracy_score(y_train2,preds))
print("F1 score : ", f1_score(y_train2,preds,average='macro'))
print(classification_report(y_train2,preds))
LABELS = ['Laying Down', 'Sitting', 'Standing', 'Walking']
confusion_mat = confusion_matrix(y_train2,preds)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();


clf3.fit(X_f, y_train_all)
preds = clf3.predict(X_f_test2)
#print(preds)
print("Acc :",accuracy_score(y_train2,preds))
print("F1 score : ", f1_score(y_train2,preds,average='macro'))
print(classification_report(y_train2,preds))
LABELS = ['Laying Down', 'Sitting', 'Standing', 'Walking']
confusion_mat = confusion_matrix(y_train2,preds)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();

clf3.fit(X_mi, y_train_all)
preds = clf3.predict(X_mi_test2)
#print(preds)
print("Acc :",accuracy_score(y_train2,preds))
print("F1 score : ", f1_score(y_train2,preds,average='macro'))
print(classification_report(y_train2,preds))
LABELS = ['Laying Down', 'Sitting', 'Standing', 'Walking']
confusion_mat = confusion_matrix(y_train2,preds)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_mat, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();


