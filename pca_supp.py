import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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
prcomp = PCA()
pipe = Pipeline([('STD',scaler),('PCA',prcomp)])
components = pipe.fit_transform(X_train)
print(np.cumsum(prcomp.explained_variance_ratio_)*100)

svm = SVC(probability=True, random_state=2022, kernel='linear')

pd_PC_trn = pd.DataFrame(components[:,:8], 
                     columns=['PC'+str(i) for i in np.arange(1,9)])

svm.fit(pd_PC_trn, y_train)

tst_comp = pipe.transform(X_test)
pd_PC_tst = pd.DataFrame(tst_comp[:,:8],
                         columns=['PC'+str(i) for i in np.arange(1,9)])

y_pred = svm.predict(pd_PC_tst)
print(accuracy_score(y_test, y_pred))

y_pred_prob = svm.predict_proba(pd_PC_tst)[:,1]
print(roc_auc_score(y_test, y_pred_prob))

################## Grid Search CV ###################
scaler = StandardScaler()
prcomp = PCA()
svm = SVC(probability=True, random_state=2022, kernel='linear')

pipe_pca_svm = Pipeline([('STD',scaler),
                         ('PCA',prcomp),('SVM',svm)])
print(pipe_pca_svm.get_params())
params = {'PCA__n_components':[0.75, 0.8, 0.85, 0.9, 0.95],
          'SVM__C':[0.4, 1, 2, 2.5]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=2022)
gcv = GridSearchCV(pipe_pca_svm, param_grid=params,
                   cv=kfold, scoring='roc_auc',verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################# hr ################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")
hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

scaler = StandardScaler()
prcomp = PCA()
svm = SVC(probability=True, random_state=2022, kernel='linear')

pipe_pca_svm = Pipeline([('STD',scaler),
                         ('PCA',prcomp),('SVM',svm)])
print(pipe_pca_svm.get_params())
params = {'PCA__n_components':[0.75, 0.8, 0.85, 0.9, 0.95],
          'SVM__C':[0.4, 1, 2, 2.5]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=2022)
gcv = GridSearchCV(pipe_pca_svm, param_grid=params,
                   cv=kfold, scoring='roc_auc',verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

