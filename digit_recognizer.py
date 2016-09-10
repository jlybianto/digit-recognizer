# ----------------
# IMPORT PACKAGES
# ----------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# ----------------
# OBTAIN DATA
# ----------------

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ----------------
# PROFILE DATA
# ----------------

train = train.as_matrix()
test = test.as_matrix()

train_target = train[0:, 0]
train_data = train[0:, 1:]

# ----------------
# MODEL AND ANALYZE DATA
# ----------------

# Support Vector Classification
clf_svc_poly = svm.SVC(cache_size=1000, class_weight='balanced', gamma=0.001, kernel='poly')
clf_svc_poly.fit(train_data, train_target)
pred_svc_poly = clf_svc_poly.predict(test)
print(Counter(pred_svc_poly))

# Random Forest Classification
clf_rfc_500 = RandomForestClassifier(n_estimators=500, oob_score=True)
clf_rfc_500.fit(train_data, train_target)
print("Out-of-Bag (OOB) Score with 500 Estimators: %f" % clf_rfc_500.oob_score_)
pred_rfc_500 = clf_rfc_500.predict(test)
print(Counter(pred_rfc_500))

clf_rfc_1000 = RandomForestClassifier(n_estimators=1000, oob_score=True)
clf_rfc_1000.fit(train_data, train_target)
print("Out-of-Bag (OOB) Score with 1000 Estimators: %f" % clf_rfc_1000.oob_score_)
pred_rfc_1000 = clf_rfc_1000.predict(test)
print(Counter(pred_rfc_1000))

# Principal Component Analysis
pca_25 = PCA(n_components=25, whiten=True)
pca_25.fit(train_data)
train_data_25 = pca_25.transform(train_data)
clf_pca_25 = svm.SVC(cache_size=1000, gamma=0.001, kernel='rbf')
clf_pca_25.fit(train_data_25, train_target)
test_25 = pca_25.transform(test)
pred_pca_25 = clf_pca_25.predict(test_25)
print(Counter(pred_pca_25))

pca_50 = PCA(n_components=50, whiten=True)
pca_50.fit(train_data)
train_data_50 = pca_50.transform(train_data)
clf_pca_50 = svm.SVC(cache_size=1000, gamma=0.001, kernel='rbf')
clf_pca_50.fit(train_data_50, train_target)
test_50 = pca_50.transform(test)
pred_pca_50 = clf_pca_50.predict(test_50)
print(Counter(pred_pca_50))

pca_100 = PCA(n_components=100, whiten=True)
pca_100.fit(train_data)
train_data_100 = pca_100.transform(train_data)
clf_pca_100 = svm.SVC(cache_size=1000, gamma=0.001, kernel='rbf')
clf_pca_100.fit(train_data_100, train_target)
test_100 = pca_100.transform(test)
pred_pca_100 = clf_pca_100.predict(test_100)
print(Counter(pred_pca_100))

pca_200 = PCA(n_components=200, whiten=True)
pca_200.fit(train_data)
train_data_200 = pca_200.transform(train_data)
clf_pca_200 = svm.SVC(cache_size=1000, gamma=0.001, kernel='rbf')
clf_pca_200.fit(train_data_200, train_target)
test_200 = pca_200.transform(test)
pred_pca_200 = clf_pca_200.predict(test_200)
print(Counter(pred_pca_200))

# ----------------
# OUTPUT DATA
# ----------------

np.savetxt("submission_svc_poly.csv", np.c_[range(1, len(test) + 1), pred_svc_poly], delimiter=",", header="ImageId,Label", comments="", fmt="%d")

np.savetxt("submission_rfc_500.csv", np.c_[range(1, len(test) + 1), pred_rfc_500], delimiter=",", header="ImageId,Label", comments="", fmt="%d")
np.savetxt("submission_rfc_1000.csv", np.c_[range(1, len(test) + 1), pred_rfc_1000], delimiter=",", header="ImageId,Label", comments="", fmt="%d")

np.savetxt("submission_pca_25.csv", np.c_[range(1, len(test) + 1), pred_pca_25], delimiter=",", header="ImageId,Label", comments="", fmt="%d")
np.savetxt("submission_pca_50.csv", np.c_[range(1, len(test) + 1), pred_pca_50], delimiter=",", header="ImageId,Label", comments="", fmt="%d")
np.savetxt("submission_pca_100.csv", np.c_[range(1, len(test) + 1), pred_pca_100], delimiter=",", header="ImageId,Label", comments="", fmt="%d")
np.savetxt("submission_pca_200.csv", np.c_[range(1, len(test) + 1), pred_pca_200], delimiter=",", header="ImageId,Label", comments="", fmt="%d")