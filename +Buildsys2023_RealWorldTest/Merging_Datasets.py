#%% importing lirbraries
import pandas as pd
import numpy as np
from datetime import datetime
from meteostat import Point, Daily, Hourly
import matplotlib.pyplot as plt
#%% Energy Analysis
DAY = 14
#%%% loading the dataset and renaming the columns
Energy_Dataset = pd.read_csv("694278-Rakhsh-1MIN.csv")
Energy_Dataset = Energy_Dataset[["Time Bucket (America/New_York)","Rakhsh-Circuit_4 (kWatts)"]]
Energy_Dataset.columns = ["time", "energy"]
Energy_Dataset["time"] = pd.to_datetime(Energy_Dataset["time"])
#%%% separating a specific day
Energy_Dataset["day"] = Energy_Dataset["time"].dt.day
Energy_Dataset["hour"] = Energy_Dataset["time"].dt.hour
Energy_Dataset["minute"] = Energy_Dataset["time"].dt.minute

Energy_Dataset = Energy_Dataset[Energy_Dataset["day"] == DAY]
Energy_Dataset = Energy_Dataset.drop(["day"], axis = 1)

# reseting the index of the rows
Energy_Dataset = Energy_Dataset.reset_index(drop = True)
#%%% Grouping by every 20 timesteps
chunk_size = 20
grouped = Energy_Dataset.groupby(Energy_Dataset.index // chunk_size)
Dataset = pd.DataFrame()
Dataset["hour"] = grouped['hour'].mean()
Dataset["minute"] = grouped['minute'].mean()
Dataset["energy"] = grouped['energy'].sum()

#########################################################
#-------------------------------------------------------#
# The final output is a downsampled energy dataset that shows energy consumption every 20 minutes"
#-------------------------------------------------------#
#########################################################

#%% outdoorTemperature
outdoorTemp = pd.read_excel("outdoorTemp.xlsx")
outdoorTemp = outdoorTemp.iloc[:,[0,1]]
outdoorTemp.columns = ["time", "temperature"]
outdoorTemp = outdoorTemp[::-1]
outdoorTemp = outdoorTemp.reset_index(drop = True)

# outdoorTemp_List = [(outdoorTemp["temperature"][0] + outdoorTemp["temperature"][1])/2]
# outdoorTemp_List += outdoorTemp["temperature"].tolist()
# outdoorTemp_List += [(outdoorTemp["temperature"][67] + outdoorTemp["temperature"][68])/2]
# outdoorTemp_List += [(outdoorTemp["temperature"][67] + outdoorTemp["temperature"][68])/2]

# Dataset["outdoorTemp"] = outdoorTemp_List


Dataset["outdoorTemp"] = outdoorTemp["temperature"][0]

#%% Ecobee Dataset
Ecobee_Dataset = pd.read_csv("July_14.csv")
Ecobee_Dataset["Date"] = pd.to_datetime(Ecobee_Dataset["Date"])
Ecobee_Dataset["day"] = Ecobee_Dataset["Date"].dt.day
Ecobee_Dataset = Ecobee_Dataset[Ecobee_Dataset["day"] == DAY]
Ecobee_Dataset = Ecobee_Dataset.drop(["day"], axis = 1)
Ecobee_Dataset = Ecobee_Dataset.reset_index(drop = True)

chunk_size = 4
grouped = Ecobee_Dataset.groupby(Ecobee_Dataset.index // chunk_size)

Dataset["indoorTemp"] = grouped["zoneAveTemp"].mean()
Dataset["setpoint"] = grouped["zoneCoolTemp"].mean()

#%% Comfort Analysis
# Importing Profiles
Profiles_Dataset = pd.read_excel("comfortProfile.xlsx")
Profiles_Dataset["Temperature"] = round(Profiles_Dataset["Temperature"],1)
# plt.plot(Profiles_Dataset["Temperature"], Profiles_Dataset["Probability15"])

Dataset["Comfort"] = None

for t in range(72):
    temp = np.round(Dataset['indoorTemp'][t],1)
    Dataset["Comfort"][t] = np.mean(Profiles_Dataset["Probability15"][Profiles_Dataset["Temperature"] == temp])

#%% Exporting the dataset
Dataset.to_excel("Dataset.xlsx")







