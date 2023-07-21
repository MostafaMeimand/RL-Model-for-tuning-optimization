#%% importing libraries
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
import warnings
import seaborn as sns
import matplotlib.font_manager
warnings.filterwarnings("ignore")

#%% Comfort Analysis
# Importing Profiles
Profiles_Dataset = pd.read_excel("comfortProfile.xlsx")
Profiles_Dataset["Temperature"] = round(Profiles_Dataset["Temperature"],1)
plt.plot(Profiles_Dataset["Temperature"], Profiles_Dataset["Probability15"])
Agents = 15

#%% Assigning comfort to each tempeprature
Dataset = pd.read_excel("3.5 - eta=1_July7/July07.xlsx",sheet_name=3)

Dataset["Comfort"] = None

for t in range(72):
    temp = np.round(Dataset['indoorTemp'][t],1)
    Dataset["Comfort"][t] = np.mean(Profiles_Dataset["Probability15"][Profiles_Dataset["Temperature"] == temp])

#%% Generating a CSV file
Dataset.to_csv("eta_1.csv")

#%% Plotting objective functions
n_segments = 4
mu = 0.5
my_pwlf = pwlf.PiecewiseLinFit(Profiles_Dataset["Temperature"], Profiles_Dataset["Probability" + str(Agents)])
breaks = my_pwlf.fit(n_segments).round(2)
y_breaks = my_pwlf.predict(breaks).round(2)

def eta_output(eta):
    
    Z_Energy = []
    Z_Comfort = []
    Obj_function = []
    Delta_u = np.arange(-5,6,0.1)
    
    for delta_u in Delta_u:
        x = (0.0237 * outdoorTemperature - 0.0690 * currentTemperature + delta_u * 0.3779 + 3.5942 + currentTemperature)
        # comfort = np.mean(Profiles_Dataset["Probability15"][Profiles_Dataset["Temperature"] == round(x,1)])
        comfort = my_pwlf.predict(x)
        Z_Comfort.append(comfort)
        
        energy = (0.1077 * outdoorTemperature - 0.1548 * currentTemperature - 0.8143 * delta_u + 4.8273)
        Z_Energy.append(energy)
                           
        Obj_function.append(energy/2.68 - eta * comfort)

    return Obj_function

#%%
currentTemperature = 76
outdoorTemperature = 69
    
fig, ax = plt.subplots()
ax.plot(np.arange(-5,6,0.1),eta_output(0), lw = 2.5)
ax.plot(np.arange(-5,6,0.1),eta_output(1), lw = 2.5)
ax.plot(np.arange(-5,6,0.1),eta_output(2), lw = 2.5)
ax.plot(np.arange(-5,6,0.1),eta_output(4), lw = 2.5)
ax.plot(np.arange(-5,6,0.1),eta_output(8), lw = 2.5)

ax.set_xlabel("delta setpoint")
ax.set_ylabel("Obj. Function Value", color = "black")
ax.legend(["0","1","2","4","8"])
ax.set_title(f"curr. temp. is {currentTemperature} and OT is {outdoorTemperature}")
# plt.savefig("obj function for hour = " + str(timestep) + ".png", dpi = 2000)

