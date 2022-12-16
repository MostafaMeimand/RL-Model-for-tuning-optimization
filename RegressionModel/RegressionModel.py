#%% importing requirements
#%%% Reading libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import gurobipy as gp
from gurobipy import GRB
import random
import pwlf
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import warnings
import math
warnings.filterwarnings("ignore")

#%%
SS_Data = pd.read_csv("in.csv")
SS_Data = SS_Data.drop(['Date/Time'], axis = 1)

###Temperature Model
SS_Data.columns = ["Outdoor temperature", "Zone temperature", "Setpoint", 
                    'Energy','Change Setpoint', 'Change Temperature']

X_1 = SS_Data[["Outdoor temperature","Zone temperature", 'Change Setpoint']]
Y_1 = SS_Data['Change Temperature']


X_train,X_test,y_train,y_test=train_test_split(X_1,Y_1,test_size=0.3,random_state=3)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_predicted = lr.predict(X_test)

MSE = np.square(np.subtract(y_test, y_predicted)).mean() 
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error of temperature model:\n")
print(RMSE)

print(lr.coef_)
print(lr.intercept_)

#%% Energy Model
X_1 = SS_Data[["Outdoor temperature","Zone temperature",'Setpoint']]
Y_1 = SS_Data['Energy']

X_train,X_test,y_train,y_test=train_test_split(X_1,Y_1,test_size=0.3,random_state=3)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_predicted = lr.predict(X_test)

MSE = np.square(np.subtract(y_test, y_predicted)).mean() 
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error of Energy model:\n")
print(RMSE)

print(lr.coef_)
print(lr.intercept_)

