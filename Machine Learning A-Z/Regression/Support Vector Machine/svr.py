import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:].values



#%%
#feature scaling
from sklearn.preprocessing import StandardScaler
scl_x = StandardScaler()
scl_y = StandardScaler()
X = scl_x.fit_transform(X)
y = scl_y.fit_transform(y)


#%%
#creation of SVR regressor
from sklearn.svm import SVR
regressor = SVR(kernel='rbf') #gaussian kernel
regressor.fit(X,y)


#inverse transforming the scaled part

y_pred = scl_y.inverse_transform((regressor.predict(scl_x.transform(np.array([[6.5]])))))



                                 
#%%
#visualizing SVR
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

'''
interpreting the data
The model which is in blue color doesn't fit CEO and its salary because CEO is considered as outlier
SVR model has some penalty parameters for outliers .CEO
Which means svr model is not fit for exponential models

'''