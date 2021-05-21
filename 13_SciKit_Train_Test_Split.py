# In[1] - Documentation
"""
Script - 13_SciKit_Train_Test_Split.py
Decription - Various method to split data
Author - Rana Pratap
Date - 2020
Version - 1.0
"""
print(__doc__)


# In[2] - Import Library and Read Data

from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
#iris = datasets.load_iris()
#x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,
#    train.size = 0.8,test.size = 0.2,random_state = 100)
#x_train.shape,y_train.shape
#x_test.shape,y_test.shape

#del(iris)
# In[3] - #sklearn.model_selection.train_test_split(*arrays, **options) -> list
x = np.arange(1, 25).reshape(12, 2)
y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
x
y

# In[4] - Split both input and output datasets with a single function call
x_train, x_test, y_train, y_test = train_test_split(x, y)
x_train
x_test
y_train
y_test

# In[5] - # Modify the code so you can choose the size of the test set and get a reproducible result:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=4, random_state=4)
x_train
x_test
y_train
y_test

# In[6] -# Split to keep the proportion of y values through the training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=4, stratify=y
)
x_train
x_test
y_train
y_test

# In[7] - # Turn off data shuffling and random split with shuffle=False:
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, shuffle=False
)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# In[8]
del(x,y)
del(x_train, x_test, y_train, y_test)



#https://realpython.com/train-test-split-python-data/