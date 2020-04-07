import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





'''

No of deaths for 

'''
corona = pd.read_csv("corona.csv",sep=",")

corona["Date"]=corona["Date"].astype('string')

X = corona.iloc[:,0:1]
y = corona.iloc[:,-2]  #independent variables

#truncating months and day from year and converting into literal days

X["Date"]=X["Date"].str.replace("(\d+)-(\d+)-(\d+)"," \\2-\\3",regex=True)
X["Date"]=X["Date"].transform(lambda x: int(x.strip()[0:2])*30+int(x.strip()[3:])-52).astype(int)

plt.plot(X,y,color="Blue")
plt.show()



from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

#polynomial regression with linear model. i.e coefficients are as they should be
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_poly,y)

plt.scatter(X, y,color="red")
plt.plot(X,regressor.predict(X_poly),color="blue")

print("The number of deaths will be {} for Wednesday, Thursday, Friday".format(regressor.predict(poly.fit_transform([[75],[76],[77]]))))




