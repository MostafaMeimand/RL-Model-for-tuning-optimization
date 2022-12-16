#%% Reading libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")

#%% plotting comfort
plt.rcParams['figure.figsize'] = 6, 4
font = {'family' : 'Helvetica',
        'weight' : 'regular',
        'size'   : 12}
plt.rc('font', **font)

Profiles_Dataset = pd.read_csv("min_profiles.csv")
Agents = 4
plt.plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents)], lw = 2.5)
plt.axhline(y = 0.5, color='g', linestyle='-.', lw = 2)

plt.axvline(x = 25.7, color='r', linestyle='--', lw = 2)
plt.axvline(x = 20.46, color='r', linestyle='--', lw = 2)
plt.xlabel("Temperature" + u'\u00b0' + 'C')
plt.ylabel("Comfort probability (%)")
plt.legend(["Thermal comfort profile", "minimum comfort therhold(" + '\u03B7' + ") = 0.5",
            "temperature bounds"], loc = "lower center")
# plt.savefig("Thermal comfort profile.pdf")

#%% Plotting results for one day
plt.rcParams['figure.figsize'] = 6, 4
font = {'family' : 'Helvetica',
        'weight' : 'regular',
        'size'   : 12}
plt.rc('font', **font)

X = pd.read_csv("X_eta=1000_day=14.csv")

fig, ax = plt.subplots()
ax.plot(X["Energy"][0:29], lw = 2.5, color = "black")
ax.set_xlabel("Hours")
ax.set_ylabel("Energy consumption", color = "black")
ax.set_xticks(range(0,23,2))
ax.set_xticklabels(range(1,25,2))
ax.set_title("\u03B7" + "=" + "1000")

ax2 = ax.twinx()
ax2.plot(X['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'], ls = "-.", color = "lightseagreen", lw = 2.5)

ax2.set_ylabel("Outdoor Temperature", color = "lightseagreen")
ax2.tick_params(colors = "lightseagreen")
plt.tight_layout()
# plt.savefig("Energy vs outdoor temperature_1000.png", dpi = 2000, )
#%% Plotting results for one day
plt.rcParams['figure.figsize'] = 6, 4
font = {'family' : 'Helvetica',
        'weight' : 'regular',
        'size'   : 12}
plt.rc('font', **font)

fig, ax = plt.subplots()
ax.set_title("\u03B7" + "=" + "1000")
ax.plot(X["Previous Temperature"][0:29], lw = 2.5, color = "black")
ax.set_xlabel("Hours")
ax.set_ylabel("Indoor temperature", color = "black")
ax.set_xticks(range(0,23,2))
ax.set_xticklabels(range(1,25,2))

ax2 = ax.twinx()
ax2.plot(X['Comfort'], ls = "-.", color = "lightseagreen", lw = 2.5)

ax2.set_ylabel("Comfort", color = "lightseagreen")
ax2.tick_params(colors = "lightseagreen")
plt.tight_layout()
# plt.savefig("Indoor temperature vs comfort_1000.png", dpi = 2000, )


#%%
##########################
##########################
##########################
##########################
##########################
#%%plotting energy productivity
plt.rcParams['figure.figsize'] = 6, 4
font = {'family' : 'Helvetica',
        'weight' : 'regular',
        'size'   : 12}
plt.rc('font', **font)

fixed_eta = pd.read_csv("Analysis/Scaled/Eta=0.1/eta.csv")
adaptive_eta = pd.read_csv("Analysis/Scaled/Eta=0.1Adaptive_std=0.8_rewardScaled/eta.csv")

fig, ax = plt.subplots()
ax.plot(fixed_eta["Efficiency"][0:29], lw = 2.5)
ax.plot(adaptive_eta["Efficiency"][0:29], lw = 3)
ax.set_xlabel("Days")
ax.set_ylabel("Energy productivity (%/kWh)")
ax.set_xticks(range(0,30,2))
ax.set_xticklabels(range(1,31,2))
ax.legend(["Fixed-" + '\u03B7',"Adaptive-" + '\u03B7'], loc = "upper center")

outdoor = []
for counter in range(1,30):
    X = pd.read_csv("Analysis/Scaled/Eta=0.1/X_eta=0_day=" + str(counter) + ".csv")
    outdoor.append(np.mean(X['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']))


ax2 = ax.twinx()
ax2.plot(outdoor, ls = "-.", color = "lightseagreen", lw = 2.5)
ax2.set_yticks(range(25,33))
ax2.set_ylabel("Outdoor temperature " + u'\u00b0' + 'C', color = "lightseagreen")
ax2.tick_params(colors = "lightseagreen")
plt.tight_layout()
# plt.savefig("EP.png", dpi = 2000)

#%%

#%% plotting comfort
fixed_comfort = []
for counter in range(1,30):
    X = pd.read_csv("Analysis/Scaled/Eta=0.1/X_eta=0_day=" + str(counter) + ".csv")
    X["Comfort"] = None
    for timestep in range(24):
        cur_temperature = X['Previous Temperature'][timestep]
        X["Comfort"][timestep] = float(Profiles_Dataset["Probability" + str(Agents)][Profiles_Dataset["Temperature"] == np.round(cur_temperature,2)])
    fixed_comfort.append(X["Comfort"].mean())
    
adaptive_comfort = []
for counter in range(1,30):
    X = pd.read_csv("Analysis/Scaled/Eta=0.1Adaptive_std=0.8_rewardScaled/X_eta=0_day=" + str(counter) + ".csv")
    X["Comfort"] = None
    for timestep in range(24):
        cur_temperature = X['Previous Temperature'][timestep]
        X["Comfort"][timestep] = float(Profiles_Dataset["Probability" + str(Agents)][Profiles_Dataset["Temperature"] == np.round(cur_temperature,2)])
    adaptive_comfort.append(X["Comfort"].mean())

# plt.rcParams['figure.figsize'] = 6, 4
# font = {'family' : 'Helvetica',
#         'weight' : 'regular',
#         'size'   : 12}
# plt.rc('font', **font)

# plt.plot(fixed_comfort[0:29], lw = 2.5) # d 1.15
# plt.plot(adaptive_comfort[0:29], lw = 2.5)
# plt.xticks(ticks = range(0,30,2), labels = range(1,31,2))
# plt.xlabel("Days")
# plt.ylabel("Thermal comfort (%)")
# plt.legend(["Fixed-" + '\u03B7',"Adaptive-" + '\u03B7'], loc = "upper center")
# plt.savefig("TC.png", dpi = 2000)

#%% plotting eta
fixed_eta = pd.read_csv("C:/Users/mosta/Desktop/RL_Adaptive model_eta/Analysis/Eta=1/eta.csv")
adaptive_eta = pd.read_csv("C:/Users/mosta/Desktop/RL_Adaptive model_eta/Analysis/Eta=Adaptive1/eta.csv")

plt.plot(fixed_eta['eta'][0:29])
plt.plot(adaptive_eta['eta'][0:29])

plt.xlabel("Days")
plt.ylabel("Eta value")

plt.tight_layout()
plt.savefig("eta.png", dpi = 2000)
#%%
Energy_fixed = []
for counter in range(1,30):
    X = pd.read_csv("Analysis/Scaled/Eta=0.1/X_eta=0_day=" + str(counter) + ".csv")
    Energy_fixed.append(np.mean(X['Energy']))

Energy_adaptive = []
for counter in range(1,30):
    X = pd.read_csv("Analysis/Scaled/Eta=0.1Adaptive_std=0.8_rewardScaled/X_eta=0_day=" + str(counter) + ".csv")
    Energy_adaptive.append(np.mean(X['Energy']))
#%%
fig, ax = plt.subplots()
ax.plot(Energy_fixed[0:29], lw = 2.5)
ax.plot(Energy_adaptive[0:29], lw = 2.5)
ax.set_xlabel("Days")
ax.set_ylabel("Energy )kWh)")
ax.set_xticks(range(0,30,2))
ax.set_xticklabels(range(1,31,2))
ax.legend(["Fixed-" + '\u03B7',"Adaptive-" + '\u03B7'], loc = "upper center")


outdoor = []
for counter in range(1,30):
    X = pd.read_csv("Analysis/Scaled/Eta=0.1/X_eta=0_day=" + str(counter) + ".csv")
    outdoor.append(np.mean(X['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']))

ax2 = ax.twinx()
ax2.plot(outdoor, ls = "-.", color = "lightseagreen", lw = 2.5)
ax2.set_yticks(range(25,33))
ax2.set_ylabel("Outdoor temperature " + u'\u00b0' + 'C', color = "lightseagreen")
ax2.tick_params(colors = "lightseagreen")
plt.tight_layout()
# plt.savefig("Energy_outdoor.png", dpi = 2000)
#%%
plt.plot(adaptive_comfort)
plt.plot(fixed_comfort)
plt.xlabel("Days")
plt.ylabel("Energy consumption(kwH)")

plt.legend(["Adaprive", "Fixed"])
plt.tight_layout()
plt.savefig("comfort.png", dpi = 2000)