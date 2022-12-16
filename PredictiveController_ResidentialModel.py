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
import torch
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

#%%% Importing electricity price
Electricity_Price = pd.read_excel("DDRC implementation.xlsx",sheet_name = "Electricity Price")
Electricity_Price = Electricity_Price["Elecricity Price"]/1000
Electricity_Price = Electricity_Price[0:96]

#Importing thermal comfort profiles
Profiles_Dataset = pd.read_csv("min_profiles.csv")

#%%
def approx_PWLF(): # approaximating objective function and constraints
    my_pwlf = pwlf.PiecewiseLinFit(Profiles_Dataset["Temperature"], Profiles_Dataset["Probability" + str(Agents)])
    breaks = my_pwlf.fit(n_segments).round(2)
    y_breaks = my_pwlf.predict(breaks).round(2)
    
    maxConstraint = Profiles_Dataset["Temperature"][Profiles_Dataset["Probability" + str(Agents)].round(2) == mu].max()
    minConstraint = Profiles_Dataset["Temperature"][Profiles_Dataset["Probability" + str(Agents)].round(2) == mu].min()
    
    return breaks, y_breaks, maxConstraint, minConstraint

def ReadExcel():
    Output = pd.read_csv("out\in.csv")
    Output = Output.drop(range(0,192))
    Output = Output.reset_index()
    Output = Output.drop(['index','LIVING_UNIT1:Zone People Occupant Count [](TimeStep)',
                         'ATTIC_UNIT1:Zone Mean Air Temperature [C](TimeStep)',
                         'ATTIC_UNIT1:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)',
                         'HTGSETP_SCH:Schedule Value [](TimeStep)',
                         'CLGSETP_SCH:Schedule Value [](TimeStep)'], axis = 1)
    Output['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) '] *= 2.77778e-7
    Output["time"] = Output.index
    return Output

def textGenerator():
    k = 0
    String = ''
    String += "IF (Hour == " + str(X["Hours"][k]) + ") && (Minute  <=  " + str(X["Minutes"][k]) + "),  Set myCLGSETP_SCH" + " = " + str(X["setpoint"][k]) + "," + "\n"
    for k in range(1,96):
        String += "ELSEIF (Hour == " + str(X["Hours"][k]) + ") && (Minute  <=  " + str(X["Minutes"][k]) + "),  Set myCLGSETP_SCH" + " = " + str(X["setpoint"][k]) + "," + "\n" 
    String += "ENDIF;"
    return String

def CoSimulation():
    text = open("in\in_main.idf").read()
    NextFile = open("in\in.IDF","wt")
    NextFile.write(text[:166851] + "\n" + textGenerator() + text[173349:])
    NextFile.close()
    os.system("RunEPlus.bat in in")

def simulation():
    for timestep in range(1,96):
        
        alphas = np.array(initial_dist_alphas.sample())
        alphas_set.append(alphas)
        
        model = gp.Model("optim")
        z = model.addVar(name="z") # value of the objective function
        x = model.addVar(name="x") # next temperature
        delta_u = model.addVar(name="delta_u",lb=-20, ub=+20)
        # setpoint of the building
        # Adding constraints
        model.addConstr(x == alphas[0] * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep] 
                        - alphas[1] * X["Previous Temperature"][timestep-1] + delta_u * alphas[2] + alphas[3] + 
                        X["Previous Temperature"][timestep-1])
    
        my_pwlf = approx_PWLF()
        model.addConstr(x <= my_pwlf[2])
        model.addConstr(x >= my_pwlf[3])
        
        # Auxilary varialbes for the second term
        x1 = model.addVar(name="x1") 
        x2 = model.addVar(name="x2")
        x3 = model.addVar(name="x3")
        x4 = model.addVar(name="x4")
        x5 = model.addVar(name="x5")
        model.addConstr(x == my_pwlf[0][0] * x1 + my_pwlf[0][1] * x2 + my_pwlf[0][2] * x3 + my_pwlf[0][3] * x4 + my_pwlf[0][4] * x5)
        model.addConstr(x1 + x2 + x3 + x4 + x5 == 1)
        model.addSOS(GRB.SOS_TYPE2, [x1, x2 , x3, x4, x5])
        
        # defining opjective function       
        model.addConstr(z == (alphas[4] * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep]
                              - alphas[5] * x 
                              - alphas[6] * (X["setpoint"][timestep-1] + delta_u) + alphas[7]) # * Electricity_Price[timestep]
                              - eta * (my_pwlf[1][0] * x1 + my_pwlf[1][1] * x2 + my_pwlf[1][2] * x3 + 
                                       my_pwlf[1][3] * x4 + my_pwlf[1][4] * x5)
                              + 10000)
        
        model.setObjective(z, GRB.MINIMIZE)
        model.optimize()
        
        X["setpoint"][timestep + 1] = k_p * round(model.getVars()[2].x,2) + X["setpoint"][timestep]
        CoSimulation()
        X["Previous Temperature"] = ReadExcel()['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)']
        X["Energy"] = ReadExcel()['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) ']
        
        cur_temperature = X['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)'][10]
        X["Comfort cost"][timestep] = float(Profiles_Dataset["Probability" + str(Agents)][Profiles_Dataset["Temperature"] == np.round(cur_temperature,1)])
    
#%% initilization
n_segments = 4 # best number based on different tests
mu = 0.50
eta = 1
k_p = 0.1
Agents = 10
cost = []
alpha_bests = []

# Different initializations for variables
# the sequence of initilizations are :
# temperature model: 1) outdoor, 2) previous temperature, 3) setpoint change, 4) intercept then =>
# Energy model:  5) outdoor, 6) current temperature, 7) setpoint change, 8) intercept    
initial_mean = [0.01, -0.01, 0.3, 1, 0.1, -0.1, -0.3, 5]
initial_std = [0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03]
initial_dist_alphas = torch.distributions.normal.Normal(torch.tensor(initial_mean).double(),
                                                        torch.tensor(initial_std).double())
alphas_set = []
#%% X initialization
X = pd.read_csv("SF_Main_96.csv")
X = X[192:]
X = X.reset_index()
X["Index"] = X.index
X["Minutes"] = X["Index"] % 4 * 15 + 15
temp = []
for i in range(0,24):
    temp.append([i] * 4)
X["Hours"] = np.reshape(temp, (1,96))[0]

X["Previous Temperature"] = None
X["Previous Temperature"][0] = 23
X["setpoint"] = 29.44
X["setpoint"][0:4] = 20.00

CoSimulation()
X["Previous Temperature"] = ReadExcel()['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)']
X["Energy"] = ReadExcel()['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) ']
X["Comfort cost"] = None
#%%
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
simulation()

#%%
temp = ReadExcel()
plt.plot(temp['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)'])
plt.plot(temp['LIVING_UNIT1:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)'])
plt.legend(["Inside temperature","Setpoint"])
# plt.plot(Electricity_Price)


