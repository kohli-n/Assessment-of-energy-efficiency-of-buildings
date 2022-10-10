#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import ensemble
from sklearn import model_selection
from sklearn import neighbors
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#import os

#path = os.path.join('c:' + os.sep, 'Users', 'nipin', 'Desktop', 'ENB_data.csv')
#df = pd.read_csv(path)

df = pd.read_csv('ENB_data.csv')
df.head()


# In[8]:


#df.corr()
import seaborn as sns

plt.figure(figsize=(10,10)) 
sns.heatmap(df.corr(), annot=True, cmap = 'RdBu', center=0);


# In[9]:


total_charges=df['Heating_Load'] + df['Cooling_Load']
df['total_charges'] = total_charges
df['charges_classes'] = pd.qcut(df.total_charges, labels = [0,1,2,3], q=4)
df.head()


# In[10]:


data = df[df.columns[0:8]]
target=df.charges_classes
X_train, X_test, y_train, y_test=train_test_split(data, target, test_size=0.20)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
#X_train_scaled
X_test_scaled = scaler.transform(X_test)


# In[11]:


#np.arange(2,51)
clf_1=neighbors.KNeighborsClassifier()
parameters_knn={'n_neighbors' : np.arange(2,51)}
grid_clf_1 = model_selection.GridSearchCV(estimator=clf_1, param_grid=parameters_knn)
grid_knn=grid_clf_1.fit(X_train_scaled, y_train)
print('The hyperparameters retained are', grid_clf_1.best_params_, "\n")

y_pred_knn=grid_clf_1.predict(X_test_scaled)
cm=pd.crosstab(y_test, y_pred_knn, rownames=['Actual'], colnames=['Predicted'])

print(cm, "\n")

score_knn=grid_clf_1.score(X_test_scaled, y_test)

print('The score for knn model is',score_knn)



# In[12]:


clf_2=svm.SVC ()
parameters_svm={'C':[0.1 , 1 , 10 , 50], 'kernel':['rbf', 'linear']}
grid_clf_2 = model_selection.GridSearchCV(estimator=clf_2, param_grid=parameters_svm)
grid_svm=grid_clf_2.fit(X_train_scaled, y_train)
#grid_clf_2.best_params_
print('The hyperparameters retained are', grid_clf_2.best_params_, '\n')

y_pred_svm=grid_clf_2.predict(X_test_scaled)
cm=pd.crosstab(y_test, y_pred_svm, rownames=['Actual'], colnames=['Predicted'])

print(cm, '\n')

score_svm=grid_clf_2.score(X_test_scaled, y_test)

print('The score for svm model is',score_svm)


# In[13]:


clf_3=ensemble.RandomForestClassifier()
parameters_rf={'max_features': ["sqrt", "log2", None],'min_samples_split': np.arange(2,31)}
grid_clf_3 = model_selection.GridSearchCV(estimator=clf_3, param_grid=parameters_rf)
grid_rf=grid_clf_3.fit(X_train_scaled, y_train)
grid_clf_3.best_params_
print('The hyperparameters retained are', grid_clf_3.best_params_, '\n')

y_pred_rf=grid_clf_3.predict(X_test_scaled)
cm=pd.crosstab(y_test, y_pred_rf, rownames=['Actual'], colnames=['Predicted'])

print(cm, '\n')

score_rf=grid_clf_3.score(X_test_scaled, y_test)

print('The score for rf model is',score_rf)


# In[17]:


#clf_1 = neighbors.KNeighborsClassifier(n_neighbors=3)
#clf_2=svm.SVC(C= 50, kernel= 'rbf')
#clf_3=ensemble.RandomForestClassifier(max_features= None, min_samples_split= 2)
from sklearn.model_selection import GridSearchCV
vc  = VotingClassifier(estimators=[('knn', clf_1), ('svm', clf_2), ('rf', clf_3)], voting='hard')
vc.fit(X_train_scaled, y_train)
y_pred = vc.predict(X_test_scaled)
cm= pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(cm, '\n')

score_vc=vc.score(X_test_scaled, y_test)

print('The score for vc model is',score_vc, '\n')


# In[ ]:




