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

#predecting the test set results
y_pred = regressor.predict(X_test)

diff = Y_test - y_pred
print(diff)

#visualizing the train set. plotting the graph
plt.scatter(X_train,Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#for test set
plt.scatter(X_test,Y_test, color='red')
#no change here because we already obtained our unique equation. The equation will be same even if we change. slope and c is constant

plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()




