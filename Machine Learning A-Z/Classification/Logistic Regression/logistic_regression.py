#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%
dataset = pd.read_csv("Social_Network_Ads.csv")

X=dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

#%%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


#%%
#apply feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#%%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#%%
y_pred = classifier.predict(X_test)

#%%
#to compare two matrices values. how many of them we got correct


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#%%

X_set,y_set = X_train, y_train
'''
numpy has this function called meshgrid. Which can be used to create a cartesian plane from one dimensional input

input two one dimensional arrays
Example : 
    x= np.linspace(-4,4,9) #gives linearly spaced numbers between -4 and 4 
    [-4,-3,-2,-1,0,1,2,3,4]
    
    y=np.linspace(-5,5,11) 
    [-5,-4,-3,-2,-1,0,1,2,3,4,5]
    
    
    
    x,y=np.meshgrid(x,y)
    #the matrix returned as X and Y have same dimensions
    #having same dimensions means
    
    X has to be repeated 11 times vertically
1st  [[-4. -3. -2. -1.  0.  1.  2.  3.  4.]
 2nd [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
 3rd [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
 4th [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
 5th [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
 6th [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
 7th [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
 8th [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
 9th [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
 10th [-4. -3. -2. -1.  0.  1.  2.  3.  4.]
 11th [-4. -3. -2. -1.  0.  1.  2.  3.  4.]]
    
    
  Y has to be repeated 9 times horizontally
    
    1st 2nd 3rd 4th 5th 6th 7th 8th 9th
 [[-5. -5. -5. -5. -5. -5. -5. -5. -5.]    
 [-4. -4. -4. -4. -4. -4. -4. -4. -4.]
 [-3. -3. -3. -3. -3. -3. -3. -3. -3.]
 [-2. -2. -2. -2. -2. -2. -2. -2. -2.]
 [-1. -1. -1. -1. -1. -1. -1. -1. -1.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
 [ 1.  1.  1.  1.  1.  1.  1.  1.  1.]
 [ 2.  2.  2.  2.  2.  2.  2.  2.  2.]
 [ 3.  3.  3.  3.  3.  3.  3.  3.  3.]
 [ 4.  4.  4.  4.  4.  4.  4.  4.  4.]
 [ 5.  5.  5.  5.  5.  5.  5.  5.  5.]]
 
 Now they act as catesian plane . Whats on -4,-5 its on the top.
 its like allowing electricity to flow from -4 wire that sits vertically
 and on -5 wire that sits horizontally
 altogether the place where thay intersect, if there was LED then it would light up
 
 
 
'''
from matplotlib.colors import ListedColormap
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop =X_set[:,0].max()+1,step=0.01)
                    ,np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01)
                    )
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

#limits the current axis

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set ==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.title('Classifier')
plt.show()

