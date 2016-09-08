# ----------------
# IMPORT PACKAGES
# ----------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier

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

# ----------------
# OUTPUT DATA
# ----------------

np.savetxt("submission_svc_poly.csv", np.c_[range(1, len(test) + 1), pred_svc_poly], delimiter=",", header="ImageId,Label", comments="", fmt="%d")
np.savetxt("submission_rfc_500.csv", np.c_[range(1, len(test) + 1), pred_rfc_500], delimiter=",", header="ImageId,Label", comments="", fmt="%d")
np.savetxt("submission_rfc_1000.csv", np.c_[range(1, len(test) + 1), pred_rfc_1000], delimiter=",", header="ImageId,Label", comments="", fmt="%d")