# This code works but learning thermal comfort is not as interesting as we thought
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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#%%% Importing electricity price
# Electricity_Price = pd.read_excel("DDRC implementation.xlsx",sheet_name = "Electricity Price")
# Electricity_Price = Electricity_Price["Elecricity Price"]/1000
# Electricity_Price = Electricity_Price[0:96]

#Importing thermal comfort profiles
Profiles_Dataset = pd.read_csv("min_profiles.csv")
Agents = 4
plt.plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents)])
# plt.plot(my_pwlf[0],my_pwlf[1])

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

def CoSimulation(month,day):
    text = open("in\in_main.idf").read()
    NextFile = open("in\in.IDF","wt")
    NextFile.write(text[:12564] + str(month) + text[12567:12610] + str(day) + text[12614:12707] + 
          str(month) + text[12711:12752] + str(day) + text[12758:166864] + 
          textGenerator() + text[168122:])
    NextFile.close()
    os.system("RunEPlus.bat in in")

def simulation(month,day):
    for timestep in range(1,24):
        model = gp.Model("optim")
        z = model.addVar(name="z") # value of the objective function
        x = model.addVar(name="x") # next temperature
        delta_u = model.addVar(name="delta_u",lb=-5, ub=+5)
        # setpoint of the building
        # Adding constraints
        model.addConstr(x == 0.0076 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep] 
                        - 0.063 * X["Previous Temperature"][timestep-1] + delta_u * 0.4392 + 1.2504 + 
                        X["Previous Temperature"][timestep-1])
            
        # model.addConstr(x <= my_pwlf[2])
        # model.addConstr(x >= my_pwlf[3])
        
        model.addConstr(x <= 28)
        model.addConstr(x >= 20)
        
        
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
        model.addConstr(z == 1/3.82 * (0.106 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep]
                              - 0.0625 * x 
                              - 0.3430 * (X["setpoint"][timestep-1] + delta_u) + 7.6881) # * Electricity_Price[timestep]
                              - 1/0.8 * float(eta["eta"][eta["counter"] == counter]) * (my_pwlf[1][0] * x1 + my_pwlf[1][1] * x2 + my_pwlf[1][2] * x3 + 
                                       my_pwlf[1][3] * x4 + my_pwlf[1][4] * x5)
                              + 10000)
        
        model.setObjective(z, GRB.MINIMIZE)
        model.optimize()
       
        X["setpoint"][timestep + 1] = k_p * round(model.getVars()[2].x,2) + X["setpoint"][timestep]
        CoSimulation(month,day)
        X["Previous Temperature"] = ReadExcel()['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)']
        X["Energy"] = ReadExcel()['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) ']

def cost_calculation(X):
    X["Comfort"] = None
    for timestep in range(24):
        cur_temperature = X['Previous Temperature'][timestep]
        X["Comfort"][timestep] = float(Profiles_Dataset["Probability" + str(Agents)][Profiles_Dataset["Temperature"] == np.round(cur_temperature,2)])
    return X["Comfort"].mean()/X["Energy"].mean()*X['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].mean()

def update_eta(eta_mean):
    eta_prime = eta
    eta_mean = eta_mean + ro * (eta_prime.sort_values("Efficiency",ascending = False)["eta"][0:n_c].mean() - eta_mean)
    return eta_mean


#%% initialization
n_segments = 4 # best number based on different tests
mu = 0.50 #minimum thershold for sacrifycing comfort
my_pwlf = approx_PWLF()
n_c = 3 # number of days to update model
k_p = 0.1 # the coefficient of changing setpoint
ro = 0.5

#%%
eta = pd.DataFrame(range(1,32), columns = ["counter"])
eta["month"] = None
eta["day"] = None

eta["month"][0:31] = np.repeat(7,31)
eta["day"][0:31] = range(1,32)


eta["Efficiency"] = None
# eta_mean = 0.1
# eta_std = 0.8
# eta_dist = torch.distributions.normal.Normal(torch.tensor(eta_mean).double(),torch.tensor(eta_std).double())

eta["eta"] = 0.5 # for fixed eta model
# eta["eta"] = None

# for i in range(1,6):
#     eta["eta"][eta["counter"] == i] = abs(float(eta_dist.sample()))

# eta["eta_mean"] = None
# eta["eta_std"] = None
# eta["eta_mean"][0:5] = eta_mean
# eta["eta_std"][0:5] = eta_std

#%% initializing the model    
for counter in range(28,30):
    X = pd.read_csv("Days/SF_main_24_" + str(counter) + ".csv")
    X = X.reset_index()
    X["Hours"] = X.index
    X["Previous Temperature"] = None
    X["Previous Temperature"][0] = 23
    X["setpoint"] = 29.44
    X["setpoint"][0:2] = 22.5
    CoSimulation(int(eta["month"][eta["counter"] == counter]),int(eta["day"][eta["counter"] == counter]))

    X["Previous Temperature"] = ReadExcel()['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)']
    X["Energy"] = ReadExcel()['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) ']
    X['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'] = ReadExcel()['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
    X["Date/Time"] = ReadExcel()['Date/Time']
    
    simulation(int(eta["month"][eta["counter"] == counter]),int(eta["day"][eta["counter"] == counter]))
    
    eta["Efficiency"][eta["counter"] == counter] = cost_calculation(X)
    
    X.to_csv("X_eta=" + str(int(eta["eta"][eta["counter"] == counter])) + "_day=" + str(counter) + ".csv")
    eta.to_csv("eta.csv")
    
#%% running the model
for counter in range(21,32):
    eta_mean = float(eta["eta_mean"][eta["counter"] == counter - 1])
    eta["eta_mean"][eta["counter"] == counter] = update_eta(eta_mean)
    eta["eta_std"][eta["counter"] == counter] = float(eta["eta_std"][eta["counter"] == counter - 1] * 0.9)
    eta_dist = torch.distributions.normal.Normal(torch.tensor(float(eta["eta_mean"][eta["counter"] == counter])).double(),torch.tensor(float(eta["eta_std"][eta["counter"] == counter])).double())
    eta["eta"][eta["counter"] == counter] = abs(float(eta_dist.sample()))
    X = pd.read_csv("Days/SF_main_24_" + str(counter) + ".csv")
    
    X = X.reset_index()
    X["Hours"] = X.index
    X["Previous Temperature"] = None
    X["Previous Temperature"][0] = 23
    X["setpoint"] = 29.44
    X["setpoint"][0:2] = 22.5
    
    CoSimulation(int(eta["month"][eta["counter"] == counter]),int(eta["day"][eta["counter"] == counter]))
    
    X["Previous Temperature"] = ReadExcel()['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)']
    X["Energy"] = ReadExcel()['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) ']
    X['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'] = ReadExcel()['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
    X["Date/Time"] = ReadExcel()['Date/Time']
    
    simulation(int(eta["month"][eta["counter"] == counter]),int(eta["day"][eta["counter"] == counter]))
    
    eta["Efficiency"][eta["counter"] == counter] = cost_calculation(X)
    
    X.to_csv("X_eta=" + str(int(eta["eta"][eta["counter"] == counter])) + "_day=" + str(counter) + ".csv")
    eta.to_csv("eta.csv")
#%%
eta = pd.read_csv("eta.csv")
#%%
for counter in range(1,31):
    X = pd.read_csv("C:/Users/mosta/Desktop/RL_Adaptive model_eta/Analysis/Scaled/Eta=0.1/X_eta=0_day=" + 
                    str(counter) + ".csv")
    eta["Efficiency"][eta["counter"] == counter] = cost_calculation(X)
#%%
eta.to_csv("eta.csv")

