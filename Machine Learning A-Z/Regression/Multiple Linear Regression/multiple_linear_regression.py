''' 
This is multiple linear regression

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1].values

dummy_dataset = pd.DataFrame(X,)
inVar= dataset.iloc[:,3].astype("category")
dummify = pd.get_dummies(inVar)
dummy_dataset.drop(dummy_dataset.columns[-1],axis=1,inplace=True)
X=pd.concat([dummy_dataset,dummify],axis=1)
cols = ['California','Florida','New York',0,1,2]
X=X[cols]


#Avoiding the dummy variable trap
X = X.iloc[:,1:]


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)



#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


y_pred = regressor.predict(X_test)

#backward elimination
import statsmodels.api as sm
#to compute p-values

#since our formula is y = bx + b1x1 + b2x2 .......we need b even though that x is always 1 . We need to know if that b has some significance or not

X = np.append(arr = np.ones((50,1)).astype(int), values=X,axis=1 )
X_opt = X[:,[0,1,2,3,4,5]].astype(int)
print(X_opt.dtype)

#select the significance variable
#ordinary least square

regressor_OLS = sm.OLS(Y,X_opt).fit()
print(regressor_OLS.summary())
#x1 is the most signigicant






