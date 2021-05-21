# In[1] - Documentation
"""
Script - 12_Scikit_Model_Metrics.py
Decription - Various predictive models and metrics
Author - Rana Pratap
Date - 2020
Version - 1.0
"""
print(__doc__)


# In[2] - Models
# Regression
## LinearRegression
from sklearn.linear_model import LinearRegression
LR_Model = LinearRegression()
print(LR_Model)

## DecisionTree
from sklearn.tree import DecisionTreeRegressor
DT_Model = DecisionTreeRegressor()
print(DT_Model)

## RandomForest
from sklearn.ensemble import RandomForestRegressor
RF_Model = RandomForestRegressor()
print(RF_Model)

## SVR
from sklearn.svm import SVR
SVR_Model = SVR()
print(SVR_Model)

# Classification
## LogisticRegression
from sklearn.linear_model import LogisticRegression
LoR_Model = LogisticRegression()
print(LoR_Model)

# In[3] - Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
y_true = [2,0,2,2,0,1]
y_pred = [1,0,2,1,0,0]

print(confusion_matrix(y_true,y_pred))

print(accuracy_score(y_true,y_pred))

print(mean_squared_error(y_true,y_pred))

print(mean_absolute_error(y_true,y_pred))

# In[] -
del(y_true,y_pred)
