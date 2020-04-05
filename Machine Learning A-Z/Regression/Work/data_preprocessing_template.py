# Data Preprocessing Template

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X, Y,test_size=1/3,random_state=0)


# Fitting linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)



