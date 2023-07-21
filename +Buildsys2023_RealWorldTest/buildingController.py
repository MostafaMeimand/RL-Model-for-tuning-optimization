#%% importing requirements
#%%% Reading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import gurobipy as gp
from gurobipy import GRB
import pwlf
import torch
import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cohort
from time import sleep
import requests
import json
from datetime import datetime, timedelta

HEAT_HOLD_TEMP = 680

#%%
#Importing thermal comfort profiles
Profiles_Dataset = pd.read_excel("comfortProfile.xlsx")
Agents = 15
plt.plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents)])
# plt.plot(my_pwlf[0], my_pwlf[1])
# plt.axhline(y = 0.5)

#%% Breaking the comfort profile into segments
def approx_PWLF(): # approaximating objective function and constraints
    n_segments = 4
    mu = 0.5
    my_pwlf = pwlf.PiecewiseLinFit(Profiles_Dataset["Temperature"], Profiles_Dataset["Probability" + str(Agents)])
    breaks = my_pwlf.fit(n_segments).round(2)
    y_breaks = my_pwlf.predict(breaks).round(2)
    
    maxConstraint = Profiles_Dataset["Temperature"][Profiles_Dataset["Probability" + str(Agents)].round(2) == mu].max()
    minConstraint = Profiles_Dataset["Temperature"][Profiles_Dataset["Probability" + str(Agents)].round(2) == mu].min()
    
    return breaks, y_breaks, maxConstraint, minConstraint

my_pwlf = approx_PWLF()

#%% Getting future temperature
def futureTemperature():
    
    API_KEY = '773d26c7dc9c8cc08f663e133833c7de'
    CITY_NAME = 'Blacksburg'
    BASE_URL = 'http://api.openweathermap.org/data/2.5/forecast'
    
    # Get the current timestamp and the timestamp 15 minutes from now
    current_time = datetime.now()
    future_time = current_time + timedelta(minutes=15)
    future_timestamp = int(future_time.timestamp())
    
    # Make the API request
    params = {
        'q': CITY_NAME,
        'appid': API_KEY,
        'units': 'metric',
    }
    response = requests.get(BASE_URL, params=params)
    data = json.loads(response.text)
    
    # Find the temperature closest to the future timestamp
    closest_temp = None
    for forecast in data['list']:
        forecast_time = int(forecast['dt'])
        if forecast_time >= future_timestamp:
            closest_temp = forecast['main']['temp'] * 1.8 + 32
            break
    
    return closest_temp

#%% Optimization model
def Optimization(currentTemperature,outdoorTemperature):
    model = gp.Model("optim")
    z = model.addVar(name="z", lb = -30) # value of the objective function
    z_energy = model.addVar(name="z_energy", lb = -30) # value of energy term
    z_comfort = model.addVar(name="z_comfort", lb = -30) # value of energy term
    x = model.addVar(name="x") # next temperature
    delta_u = model.addVar(name="delta_u",lb=-2, ub=+2)
    # setpoint of the building
    # Temperature predictive model
    model.addConstr(x == 0.0237 * outdoorTemperature - 0.0690 * currentTemperature + delta_u * 0.3779 + 3.5942 + currentTemperature)
    
    #Temperature constraint
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
    
    # defining opjective functions      
    model.addConstr(z_comfort == (my_pwlf[1][0] * x1 + my_pwlf[1][1] * x2 + my_pwlf[1][2] * x3 + 
                                   my_pwlf[1][3] * x4 + my_pwlf[1][4] * x5))
    model.addConstr(z_energy ==  0.1077 * outdoorTemperature - 0.1548 * currentTemperature - 0.8143 * delta_u + 4.8273)
    
    model.addConstr(z == z_energy/2.68 - eta * z_comfort)
                    
    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()

    return model.getVars()[3].x, model.getVars()[4].x

#%% controlling based on setpoint
ecobee = cohort.Ecobee()
ecobee.setHold(HEAT_HOLD_TEMP, 750)
eta = 4.8

#%%
while(True):
    ecobee = cohort.Ecobee()
    curr_temp = ecobee.getData().get('runtime').get("actualTemperature")
    outdoorTemperature = futureTemperature()
    opt_setpoint = Optimization(curr_temp/10, outdoorTemperature)
    
    print("------------------------")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    
    print("------------------------")
    print("current temperature is:", curr_temp)
    
    print("------------------------")
    print("Outdoor temperature is:", outdoorTemperature)
    
    print("------------------------")
    print("Recommended temperature is:", opt_setpoint[0])
    
    print("------------------------")
    print("Thermostat change is:", opt_setpoint[1])
    
    finalSetpoint = int(ecobee.getData().get('runtime').get("desiredCool") + opt_setpoint[1] * 10)
    
    if finalSetpoint > 780:
        finalSetpoint = 780
        
    if finalSetpoint < 700:
        finalSetpoint = 700
    
    print("------------------------")
    print("Final setpoint is:", finalSetpoint)
    ecobee.setHold(HEAT_HOLD_TEMP, finalSetpoint)
    
    print(ecobee.getData().get('runtime'))
    sleep(20 * 60)

#%%
# # Setting random setpoint
# opt_temp = round(random.random() * 70 + 710)
# ecobee.setHold(HEAT_HOLD_TEMP, 760)
# print(opt_temp)
# # current date time printing

#%%
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################
#%%
# def cost_calculation(X):
#     X["Comfort"] = None
#     for timestep in range(24):
#         cur_temperature = X['Previous Temperature'][timestep]
#         X["Comfort"][timestep] = float(Profiles_Dataset["Probability" + str(Agents)][Profiles_Dataset["Temperature"] == np.round(cur_temperature,2)])
#     return X["Comfort"].mean()/X["Energy"].mean()

# def update_eta(eta_mean):
#     eta_prime = eta
#     eta_mean = eta_mean + ro * (eta_prime.sort_values("Efficiency",ascending = False)["eta"][0:n_c].mean() - eta_mean)
#     return eta_mean

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
eta_mean = 1
eta_std = 4
eta_dist = torch.distributions.normal.Normal(torch.tensor(eta_mean).double(),torch.tensor(eta_std).double())

eta["eta"] = 1 # for fixed eta model
# eta["eta"] = None # for fixed eta model

# for i in range(1,6):
#     eta["eta"][eta["counter"] == i] = abs(float(eta_dist.sample()))

eta["eta_mean"] = None
eta["eta_mean"]=1#[0:5] = 1

#%% initializing the model    
for counter in range(1,6):
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
    eta["eta_mean"][eta["counter"] == counter] = eta_mean
    
    X.to_csv("X_eta=" + str(int(eta["eta"][eta["counter"] == counter])) + "_day=" + str(counter) + ".csv")
    eta.to_csv("eta.csv")
    
#%% running the model
for counter in range(28,32):
    
    eta_mean = float(eta["eta_mean"][eta["counter"] == counter - 1])
    eta["eta_mean"][eta["counter"] == counter] = update_eta(eta_mean)
    eta_std = eta_std * 0.9
    eta_dist = torch.distributions.normal.Normal(torch.tensor(float(eta["eta_mean"][eta["counter"] == counter])).double(),torch.tensor(eta_std).double())
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
    
    # eta["Efficiency"][eta["counter"] == counter] = cost_calculation(X)
    
    X.to_csv("X_eta=" + str(int(eta["eta"][eta["counter"] == counter])) + "_day=" + str(counter) + ".csv")
    eta.to_csv("eta.csv")
