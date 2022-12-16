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
Agents = 4
plt.plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents)])
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
    Output = Output.drop(range(0,48))
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
    String += "IF (Hour == " + str(X["Hours"][k]) + "),  Set myCLGSETP_SCH" + " = " + str(X["setpoint"][k]) + "," + "\n"
    for k in range(1,24):
        String += "ELSEIF (Hour == " + str(X["Hours"][k]) + "),  Set myCLGSETP_SCH" + " = " + str(X["setpoint"][k]) + "," + "\n" 
    String += "ENDIF;"
    return String

def CoSimulation(day):
    text = open("in\in_main.idf").read()
    NextFile = open("in\in.IDF","wt")
    NextFile.write(text[:12608] + str(day) + text[12611:12746] + str(day) + text[12749:166855] +
                   textGenerator() + text[173354:])
    NextFile.close()
    os.system("RunEPlus.bat in in")

def simulation(day):
    for timestep in range(1,24):
        
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
        CoSimulation(day)
        X["Previous Temperature"] = ReadExcel()['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)']
        X["Energy"] = ReadExcel()['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) ']
#%% initialization
n_segments = 5 # best number based on different tests
mu = 0.50
eta = 5
k_p = 0.1
# n_c = 5
# ro = 0.5
#%% running model
    X = pd.read_csv("SF_Main_24.csv")
    X = X[48:]
    X = X.reset_index()
    X["Hours"] = X.index
    X["Previous Temperature"] = None
    X["Previous Temperature"][0] = 23
    X["setpoint"] = 29.44
    X["setpoint"][0:2] = 22.5
    
    CoSimulation(1)
    X["Previous Temperature"] = ReadExcel()['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)']
    X["Energy"] = ReadExcel()['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) ']

