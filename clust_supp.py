import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")

brupt = pd.read_csv("Bankruptcy.csv", index_col=0)
X = brupt.drop(['D', 'YR'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)

scaler = StandardScaler()
X_trn_scl = scaler.fit_transform(X_train)
# Finding the best cluster based on Silhouette
sil = []
for i in np.arange(2,10):
    km = KMeans(n_clusters=i, random_state=2022)
    km.fit(X_trn_scl)
    labels = km.predict(X_trn_scl)
    sil.append(silhouette_score(X_trn_scl, labels))

Ks = np.arange(2,10)
i_max = np.argmax(sil)
best_k = Ks[i_max]
print("Best K =", best_k)

scaler = StandardScaler()
km = KMeans(n_clusters=best_k, random_state=2022)
pipe = Pipeline([('STD',scaler),('KM',km)])
pipe.fit(X_train)
labels = pipe.predict(X_train)

X_train['Cluster'] = labels
X_train['Cluster'] = X_train['Cluster'].astype('category')

X_trn_ohe = pd.get_dummies(X_train)

labels = pipe.predict(X_test)

X_test['Cluster'] = labels
X_test['Cluster'] = X_test['Cluster'].astype('category')
X_tst_ohe = pd.get_dummies(X_test)

# Removing Cluster_3 as it is not present in X_test
X_trn_ohe.drop('Cluster_3', axis=1, inplace=True)
rf = RandomForestClassifier(random_state=2022)
rf.fit(X_trn_ohe, y_train)

y_pred = rf.predict(X_tst_ohe)
print(accuracy_score(y_test, y_pred))

y_pred_prob = rf.predict_proba(X_tst_ohe)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

###################One Hot Encoder#########################
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer,make_column_selector
ohc = OneHotEncoder()


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)


labels = pipe.predict(X_train)
X_train['Cluster'] = labels
X_train['Cluster'] = X_train['Cluster'].astype('object')

ct = make_column_transformer((ohc,
       make_column_selector(dtype_include=object)),
                             ("passthrough",
                              make_column_selector(dtype_include='float64')))
dum_np = ct.fit_transform(X_train)

print(ct.get_feature_names_out())

########################################################################
