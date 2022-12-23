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

#%% Comfort profile
Profiles_Dataset = pd.read_csv("min_profiles.csv")
Agents = 4

def cost_calculation(X):
    X["Comfort"] = None
    for timestep in range(24):
        cur_temperature = X['Previous Temperature'][timestep]
        X["Comfort"][timestep] = float(Profiles_Dataset["Probability" + str(Agents)][Profiles_Dataset["Temperature"] == np.round(cur_temperature,2)])
    return X["Comfort"].mean()/X["Energy"].mean()*X['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].mean()

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










