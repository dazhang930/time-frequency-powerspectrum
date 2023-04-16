from scipy.io import loadmat
from torch.utils.data import random_split
import numpy as np
import random
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split

annots = loadmat('ls_animal.mat')
ls_animal = annots['ls_animal']
# ls_animal = ls_animal[:,:13,:]
annots = loadmat('ls_person.mat')
ls_person = annots['ls_person']
# ls_person = ls_person[:,:13,:]
annots = loadmat('ls_tool.mat')
ls_tool = annots['ls_tool']

ls_animal = ls_animal.reshape(ls_animal.shape[0], ls_animal.shape[1]*ls_animal.shape[2])
ls_person = ls_person.reshape(ls_person.shape[0], ls_person.shape[1]*ls_person.shape[2])
ls_tool = ls_tool.reshape(ls_tool.shape[0], ls_tool.shape[1]*ls_tool.shape[2])

print(ls_animal.shape)
print(ls_person.shape)
print(ls_tool.shape)

x = np.concatenate([ls_animal, ls_person, ls_tool])
y = np.concatenate([ [0]*ls_animal.shape[0], [1]*ls_person.shape[0], [2]*ls_tool.shape[0] ])


# paths = np.concatenate([ls_animal, ls_tool])


# x = np.concatenate([ls_person, paths])
# y = np.concatenate([ [1]*ls_person.shape[0], [0]*paths.shape[0] ])

paths = np.concatenate([ls_animal, ls_tool])
# paths = np.concatenate([ls_person, ls_tool])

x = np.concatenate([ls_person, paths])
# x = np.concatenate([ls_animal, paths])

y = np.concatenate([ [1]*ls_person.shape[0], [0]*paths.shape[0] ])
# y = np.concatenate([ [1]*ls_animal.shape[0], [0]*paths.shape[0] ])

cv = LeaveOneOut()
# enumerate splits
y_true, y_pred = list(), list()
for train_ix, test_ix in cv.split(x):
 # split data
 x_train, x_test = x[train_ix, :], x[test_ix, :]
 y_train, y_test = y[train_ix], y[test_ix]
 # fit model
#  print(y_test)
 model = svm.SVC(random_state=1, kernel='linear', probability=True)
#  model = KNeighborsClassifier(n_neighbors=10, weights='distance')
 model.fit(x_train, y_train)
 # evaluate model

 yhat = model.predict_proba(x_test)[::,1]
#  yhat = model.predict(x_test)
 # store
#  print(y_test, yhat)
 y_true.append(y_test)
 y_pred.append(yhat)

# fpr2, tpr2, _ = metrics.roc_curve(y_test,  yhat)
# print(fpr2, tpr2)
fpr2, tpr2, _ = metrics.roc_curve(y_true, y_pred)

auc2 = metrics.roc_auc_score(y_true, y_pred)

plt.plot(fpr2,tpr2,label="SVM Person, (AUC = %0.3f)"%(auc2))
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier') 
plt.legend(loc=4)
plt.show()



cv = LeaveOneOut()
# enumerate splits
y_true, y_pred = list(), list()
for train_ix, test_ix in cv.split(x):
 # split data
 x_train, x_test = x[train_ix, :], x[test_ix, :]
 y_train, y_test = y[train_ix], y[test_ix]
 # fit model
#  print(y_test)
 model = svm.SVC(random_state=1, kernel='linear', probability=True)
 model.fit(x_train, y_train)
 # evaluate model

#  yhat = model.predict_proba(x_test)[::,1]
 yhat = model.predict(x_test)
 # store
#  print(y_test, yhat)
 y_true.append(y_test)
 y_pred.append(yhat)

# fpr2, tpr2, _ = metrics.roc_curve(y_test,  yhat)
# print(fpr2, tpr2)

# y_pred2 = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
y_pred2 = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

fpr2, tpr2, _ = metrics.roc_curve(y_true, y_pred)
auc2 = metrics.roc_auc_score(y_true, y_pred)

fpr3, tpr3, _ = metrics.roc_curve(y_true, y_pred2)
auc3 = metrics.roc_auc_score(y_true, y_pred2)

print("AUC_ROC Score: ", auc2, auc3)

plt.plot(fpr3,tpr3,color='orange', label="CNN Person, (AUC = %0.3f)"%(auc3))
plt.plot(fpr2,tpr2,color='blue', label="SVM Person, (AUC = %0.3f)"%(auc2))
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier') 
plt.legend(loc=4)
plt.show()


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, average_precision_score

precision1, recall1, thresholds = precision_recall_curve(y_true, y_pred)
precision2, recall2, thresholds2 = precision_recall_curve(y_true, y_pred2)
# print(len(precision1), len(recall1))

prauc = round(auc(recall1, precision1),6)
prauc2 = round(auc(recall2, precision2),6)
print("PR Score: ", prauc, prauc2)

y_true = np.asarray(y_true)
baseline = len(y_true[y_true==1]) / len(y_true)

plt.plot(recall2,precision2,color='orange', label="CNN Person, (AUC = %0.3f)"%(prauc2))
plt.plot(recall1,precision1,color='blue', label="SVM Person, (AUC = %0.3f)"%(prauc))
plt.plot([0, 1], [baseline, baseline], color='red', linestyle='--', label='Baseline')
plt.legend(loc=4)
plt.show()

