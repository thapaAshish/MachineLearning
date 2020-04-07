import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Interpretation of Dataset we have in our hand
We are given position with its level and salary related to it.

Levels are integers.
our job is to predict the salary of someone with level 6.5 


If graph is drawn taking position as X axis and salary as Y axis 
then exponential curve will be seen. Which means salary increases exponentially as position level is increased

'''

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)


'''
polynomial features
can give x^2 x^3 .... for given array
'''
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(3)
X_poly = poly_reg.fit_transform(X)

'''
Fitting the Linear regressor with polynomial features
The name is polynomial but why linear regression model is being used.
The trick is to focus on coefficients rather than the variables here


a real exponential regressor must have its coefficients varied exponentially

'''
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)
'''
This is linear regression model being plotted
here real data is scattered in red color
Here no need to scale because the module we are importing takes care of scaling

'''
plt.scatter(X,y,color='red')
plt.plot(X,linreg.predict(X),color="blue")
plt.title("Linear regression model")
plt.xlabel("Position_Level")
plt.ylabel("Salary")
plt.show()

'''
Now drawing the graph of polynomial regression
which means predicting too.
Notice that we are plotting X and X_poly

'''
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(X_poly),color="blue")
plt.title("Linear regression model")
plt.xlabel("Position_Level")
plt.ylabel("Salary")
plt.show()

