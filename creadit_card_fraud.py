# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:09:36 2021

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snp
fraud_org=pd.read_csv('C:/Users/user/.spyder-py3/creditcard.csv')
df2=fraud_org.loc[fraud_org['Class']==0].sample(n=492)
df3=fraud_org.loc[fraud_org['Class']==1]
df4=pd.concat([df2,df3], ignore_index=True)
from scipy import stats
import numpy as np
z=np.abs(stats.zscore(df4))
df4=df4[(z<3).all(axis=1)]

y=df4['Class']
x=df4[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30)

#cross validation kneighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn_c1=KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn_c1,x,y,cv=5,scoring='accuracy').mean()


#cross validation LOGISTIC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
lg_reg=LogisticRegression()
cross_val_score(lg_reg,x,y,cv=5,scoring='accuracy').mean()


#cross validation RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rfa=RandomForestClassifier(n_estimators=20)
cross_val_score(rfa,x,y,cv=5,scoring='accuracy').mean()

#cross validation LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
lda= LinearDiscriminantAnalysis()
cross_val_score(lda,x,y,cv=5,scoring='accuracy').mean()

#cross validation Naive bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
nb= GaussianNB()
cross_val_score(nb,x,y,cv=5,scoring='accuracy').mean()

#cross validation SVM
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
sv= SVC(kernel='poly')
cross_val_score(sv,x,y,cv=5,scoring='accuracy').mean()

#cross validation RF
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=20,random_state=92)
rf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
y_predictedrf=rf.predict(x_test)
accuracy_score(y_test,y_predictedrf)

y=df4['Class']
x1=df4[['V4','V10','V12','V14','V16']]

from sklearn.model_selection import train_test_split
X1_train,X1_test,Y_train,Y_test=train_test_split(x1,y,test_size=0.2,random_state=92)

from sklearn.linear_model import LogisticRegression
lg2=LogisticRegression()
lg2.fit(X1_train,Y_train)

y_pred=lg2.predict(X1_test)

#accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)

#confusion martirx
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_predictedrf)

