#%% importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import random
import pwlf
import warnings
import seaborn as sns
import matplotlib.font_manager
warnings.filterwarnings("ignore")

#%% Plotting thermal comfort profile

plt.rcParams['figure.figsize'] = 6, 4
font = {'family' : 'Century Gothic',
        'weight' : 'regular',
        'size'   : 12}
plt.rc('font', **font)

Profiles_Dataset = pd.read_excel("comfortProfile.xlsx")

plt.plot(Profiles_Dataset["Temperature"], Profiles_Dataset["Probability15"], lw = 2.5)
plt.fill(Profiles_Dataset["Temperature"], Profiles_Dataset["Probability15"], alpha = 0.25)

plt.xlabel("Temperature")
plt.xlim([68,80])
plt.xticks(range(68,81))
plt.axvline(x = 70.6, ls = "-.", color = "r", lw = 2)
plt.axvline(x = 77.1, ls = "-.", color = "r", lw = 2)
plt.axvline(x = 73.9, ls = "-.", color = "gray", lw = 2)

plt.ylabel("Comfort probability")
plt.yticks(np.arange(0,1.1,0.1))
plt.axhline(y = 0.5, ls = "-.", color = "g", lw = 2)

plt.legend(["thermal comfort profile", "temperature boundries", "_Hidden label","thermal preference",
            "minimum comfort threshold"])

# plt.savefig("thermalComfortProfile.pdf")

#%% Plotting all measures on one graph

plt.rcParams['figure.figsize'] = 6, 4
font = {'family' : 'Century Gothic',
        'weight' : 'regular',
        'size'   : 12}
plt.rc('font', **font)

Energy = [2.13,2.09,2.90,1.90,1.81, 1.82]
Comfort = [0.79,0.95,0.93,0.97,0.96, 0.96]
Productivity = [0.37,0.46,0.32,0.51,0.53,0.52]

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)

twin1 = ax.twinx()
twin2 = ax.twinx()

# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
twin2.spines.right.set_position(("axes", 1.2))

p1, = ax.plot(range(6), Energy, "C0", label="Energy", lw = 2.5)
p2, = twin1.plot(range(6), Comfort, "C1", label="Comfort", lw = 2.5)
p3, = twin2.plot(range(6), Productivity, "C2", label="Productivity", lw = 2.5)

ax.set(ylabel="Energy")
ax.set(xlabel="Time")
ax.set_xticks(range(6))
ax.set_xticklabels(["baseline", "day-1","day-2", "day-3","day-4", "day-5"])
twin1.set(ylim = (0.76,1),ylabel="Comfort")
twin2.set(ylim = (0.3,0.6),ylabel="Productivity")

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
twin2.yaxis.label.set_color(p3.get_color())

ax.tick_params(axis='y', colors=p1.get_color())
twin1.tick_params(axis='y', colors=p2.get_color())
twin2.tick_params(axis='y', colors=p3.get_color())

ax.legend(handles=[p1, p2, p3], loc = "lower right")

plt.savefig("comfortEnergyProductivity.pdf")

#%%





