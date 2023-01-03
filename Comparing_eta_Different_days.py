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


#%%
eta = pd.read_csv("Analysis/Scaled/Eta=0.1Adaptive_std=0.8/eta.csv")
plt.plot(eta["eta"],'o-')
#%% Check eta for a single day
X = pd.read_csv("Analysis/Scaled/Eta=10/X_eta=10_day=11.csv")
X["Comfort"] = None
for timestep in range(24):
    cur_temperature = X['Previous Temperature'][timestep]
    X["Comfort"][timestep] = float(Profiles_Dataset["Probability" + str(Agents)][Profiles_Dataset["Temperature"] == np.round(cur_temperature,2)])

print(cost_calculation(X))
#%%
fig, ax = plt.subplots()
ax.plot(X['Previous Temperature'], lw = 2.5, color = "black")
ax.set_xlabel("days")
ax.set_ylabel("Eenergy consumption", color = "black")
# ax.set_xticks(range(0,23,2))
# ax.set_xticklabels(range(1,25,2))
# ax.set_title("\u03B7" + "=" + "1000")

ax2 = ax.twinx()
ax2.plot(X['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'], ls = "-.", color = "lightseagreen", lw = 2.5)
ax2.set_ylabel("Outdoor temperature", color = "lightseagreen")
ax2.tick_params(colors = "lightseagreen")
plt.tight_layout()

#%% Check the value of objective function for different values of eta
my_pwlf = approx_PWLF()

month= 7
day = 11


X = pd.read_csv("Days/SF_main_24_" + str(day) + ".csv")

X = X.reset_index()
X = ReadExcel()
X["Hours"] = X.index
X["Previous Temperature"] = None
X["Previous Temperature"][0] = 23
X["setpoint"] = 29.44
X["setpoint"][0:2] = 22.5
CoSimulation(month,day)

X["Previous Temperature"] = ReadExcel()['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)']
X["Energy"] = ReadExcel()['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) ']


eta = 1

X["Comfort"] = None

#%%%
Obj_function = []
Inside_Temperature = []
Delta_u = []
Z_Energy = []
Z_Comfort = []

# for timestep in range(1,8):
    #%%
    model = gp.Model("optim")
    z = model.addVar(name="z") # value of the objective function
    z_energy = model.addVar(name="z_energy",lb = -10) # value of the objective function for energy term
    z_comfort = model.addVar(name="z_comfort") # value of the objective function for comfort term

    x = model.addVar(name="x") # next temperature
    delta_u = model.addVar(name="delta_u",lb=-10, ub=+10)
    # setpoint of the building
    # Adding constraints
    model.addConstr(x == 1.5475 + 0.0111 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep] 
                        - 0.0799 * X["Previous Temperature"][timestep]
                         + 0.4456 * delta_u + X["Previous Temperature"][timestep-1])
    
    model.addConstr(x <= 28)
    model.addConstr(x >= 18)
    
    
    # Auxilary varialbes for the second term
    x1 = model.addVar(name="x1") 
    x2 = model.addVar(name="x2")
    x3 = model.addVar(name="x3")
    x4 = model.addVar(name="x4")
    x5 = model.addVar(name="x5")
    model.addConstr(my_pwlf[0][0] * x1 + my_pwlf[0][1] * x2 + my_pwlf[0][2] * x3 + my_pwlf[0][3] * x4 + my_pwlf[0][4] * x5 == x)
    model.addConstr(x1 + x2 + x3 + x4 + x5 == 1)
    model.addSOS(GRB.SOS_TYPE2, [x1, x2 , x3, x4, x5])
    
    # defining opjective function
    model.addConstr(z_energy == 8.0323 + 0.1203 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep]
                               - 0.1256 * x
                               - 0.3143 * (X["setpoint"][timestep-1] + delta_u))

    model.addConstr(z_comfort == my_pwlf[1][0] * x1 + my_pwlf[1][1] * x2 + my_pwlf[1][2] * x3 +
                    my_pwlf[1][3] * x4 + my_pwlf[1][4] * x5)
    
    model.addConstr(z ==  z_energy/3.72 - eta * z_comfort/0.85 + 1000)
    
    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()
    #%%
    X["setpoint"][timestep + 1] = k_p * round(model.getVars()[4].x,2) + X["setpoint"][timestep]
    CoSimulation(month,day)
    X["Previous Temperature"] = ReadExcel()['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)']
    X["Energy"] = ReadExcel()['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) ']
    
    # comfort_calculation()
    
    
    Obj_function.append(model.getVars()[0].x)
    Z_Energy.append(model.getVars()[1].x)
    Z_Comfort.append(model.getVars()[2].x)
    Inside_Temperature.append(model.getVars()[3].x)
    Delta_u.append(model.getVars()[4].x)

#%% Plotting objective function at different timesteps
def eta_output(eta):
    Z_Energy = []
    Z_Comfort = []
    Obj_function = []
    Delta_u = np.arange(-5,6,0.1)
    
    
    for delta_u in Delta_u:
        x = (1.5475 + 0.0111 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep] 
                        - 0.0799 * X["Previous Temperature"][timestep]
                         + 0.4456 * delta_u + X["Previous Temperature"][timestep-1])
        comfort = float(Profiles_Dataset["Probability" + str(Agents)][Profiles_Dataset["Temperature"] == np.round(x,2)])
        Z_Comfort.append(comfort)
        
        energy = (8.0323 + 0.1203 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep]
                               - 0.1256 * x
                               - 0.3143 * (X["setpoint"][timestep-1] + delta_u))
        Z_Energy.append(energy)                              
        Obj_function.append(energy/3.72 - eta * comfort/0.85)
    
    return Obj_function

def comfort_calculation():
    for timestep in range(24):
        cur_temperature = X['Previous Temperature'][timestep]
        X["Comfort"][timestep] = float(Profiles_Dataset["Probability" + str(Agents)][Profiles_Dataset["Temperature"] == np.round(cur_temperature,2)])


#%%
fig, ax = plt.subplots()
ax.plot(np.arange(-5,6,0.1),eta_output(0), lw = 2.5)
ax.plot(np.arange(-5,6,0.1),eta_output(0.5), lw = 2.5)
ax.plot(np.arange(-5,6,0.1),eta_output(1), lw = 2.5)
ax.plot(np.arange(-5,6,0.1),eta_output(1.5), lw = 2.5)
ax.plot(np.arange(-5,6,0.1),eta_output(2), lw = 2.5)
ax.plot(np.arange(-5,6,0.1),eta_output(4), lw = 2.5)


ax.set_xlabel("delta setpoint")
ax.set_ylabel("Obj. Function Value", color = "black")
ax.set_title("value of objective function for hour = " + str(timestep))
ax.legend(["0","0.5","1","1.5","2","4"])
plt.savefig("obj function for hour = " + str(timestep) + ".png", dpi = 2000)

#%% plotting for a single timestep
fig, ax = plt.subplots()
ax.plot(Delta_u,Z_Comfort, lw = 2.5, color = "black")
ax.set_xlabel("delta setpoint")
ax.set_ylabel("Comfort Obj. Value (negative)", color = "black")
ax.set_title("\u03B7" + "=" + str(eta))

ax2 = ax.twinx()
ax2.plot(Delta_u,Z_Energy, ls = "-.", color = "lightseagreen", lw = 2.5)
ax2.set_ylabel("Energy Obj. Value", color = "lightseagreen")
ax2.tick_params(colors = "lightseagreen")
plt.tight_layout()
ax.legend(["Comfort term", "Objective function"])
ax2.legend(["Energy term"], loc = "upper left")
# plt.savefig("obj function eta=" + str(eta) + ".png", dpi = 2000)
#%% plotting summation of two terms
fig, ax = plt.subplots()
ax.plot(Delta_u,Obj_function, lw = 2.5, color = "blue")
ax.set_xlabel("delta setpoint")
ax.set_ylabel("Obj. Function Value", color = "black")
ax.set_title("\u03B7" + "=" + str(eta))


#%% plotting objective values for a day
fig, ax = plt.subplots()
ax.plot(Z_Comfort, lw = 2.5, color = "black")
ax.set_xlabel("Hours")
ax.set_ylabel("Comfort Obj. Value", color = "black")
ax.set_xticks(range(0,23,2))
ax.set_xticklabels(range(1,25,2))
ax.set_title("\u03B7" + "=" + str(eta))

ax2 = ax.twinx()
ax2.plot(Z_Energy, ls = "-.", color = "lightseagreen", lw = 2.5)
ax2.set_ylabel("Energy Obj. Value", color = "lightseagreen")
ax2.tick_params(colors = "lightseagreen")
plt.tight_layout()
plt.savefig("obj function eta=" + str(eta) + ".png", dpi = 2000)
#%% plotting objective values for a day
fig, ax = plt.subplots()
ax.plot(X["Previous Temperature"], lw = 2.5, color = "black")
ax.set_xlabel("Hours")
ax.set_ylabel("indoor temperature", color = "black")
ax.set_xticks(range(0,23,2))
ax.set_xticklabels(range(1,25,2))
ax.set_title("\u03B7" + "=" + str(eta))

ax2 = ax.twinx()
ax2.plot(X["Energy"], ls = "-.", color = "lightseagreen", lw = 2.5)
ax2.set_ylabel("Energy", color = "lightseagreen")
ax2.tick_params(colors = "lightseagreen")
plt.tight_layout()
plt.savefig("EnergyPlus eta=" + str(eta) + ".png", dpi = 2000)
