#%%
import pandas as pd
import numpy as np

#%%
#lets import the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
#seperating dataset into two components
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,[4]].values

#%%
#splitting the dataset
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

#%%
#feature scaling because not all the algoritms scale features for you
from sklearn.preprocessing import StandardScaler

StS = StandardScaler()
X_train = StS.fit_transform(X_train)
X_test = StS.transform(X_test)

#%%
from sklearn.svm import SVC

classifier = SVC(kernel='poly')
classifier.fit(X_train,y_train.ravel())

#%%
y_pred = classifier.predict(X_test)


#%% 
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


