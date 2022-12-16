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
from fitter import Fitter, get_common_distributions, get_distributions
from scipy.stats import beta
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#%%% Importing electricity price
# Electricity_Price = pd.read_excel("DDRC implementation.xlsx",sheet_name = "Electricity Price")
# Electricity_Price = Electricity_Price["Elecricity Price"]/1000
# Electricity_Price = Electricity_Price[0:96]

#%% Generic profile and estimating the profile
Generic = pd.read_csv("Generic&None.csv")
Generic["Count"] = Generic["probability"].round(2) * 100
Empty_Generic = np.repeat(Generic['temperature'],Generic['Count'])
# G_ae, G_loce, G_scalee = stats.skewnorm.fit(Empty_Generic)

f = Fitter(Empty_Generic,distributions=['gamma','lognorm',"beta","burr","norm"])                          
f.fit()
f.summary()
Generic_best = f.get_best(method = 'sumsquare_error')

#plotting the distribution
Y1 = beta.pdf(np.linspace(18,30,1200), Generic_best['beta']['a'],Generic_best['beta']['b'],Generic_best['beta']['loc'], 
              Generic_best['beta']['scale'])
# plt.plot(np.linspace(18,30,1200), Y1)
#%% individual profiles
Profiles_Dataset = pd.read_csv("min_profiles.csv")
Agents = 28
Profiles_Dataset = Profiles_Dataset[["Temperature","Probability" + str(Agents)]]
plt.plot(Profiles_Dataset["Temperature"],Profiles_Dataset["Probability" + str(Agents)])
#%%
Profiles_Dataset["Count"] = Profiles_Dataset["Probability" + str(Agents)].round(2) * 100
Empty_profile = np.repeat(Profiles_Dataset["Temperature"],Profiles_Dataset["Count"])

# I_ae, I_loce, I_scalee = stats.skewnorm.fit(Empty_profile)
f = Fitter(Empty_profile,distributions=['gamma','lognorm',"beta","burr","norm"])                          
f.fit()
f.summary()
Ind_best = f.get_best(method = 'sumsquare_error')

Y2 = beta.pdf(np.linspace(18,30,1200), Ind_best['beta']['a'],Ind_best['beta']['b'],Ind_best['beta']['loc'], 
              Ind_best['beta']['scale'])
#%%
plt.plot(np.linspace(18,30,1200),beta.pdf(np.linspace(18,30,1200), Generic_best['beta']['a'],Generic_best['beta']['b'],Generic_best['beta']['loc'], 
              Generic_best['beta']['scale']))
plt.plot(breaks, y_breaks)
#%%
n_segments = 5
mu = 0.5
def approx_PWLF(a,b,loc,scale): # approaximating objective function and constraints
    X = np.linspace(18,30,1200)
    Y = beta.pdf(X, a, b,loc,scale)
    Y = Y/Y.max()
    my_pwlf = pwlf.PiecewiseLinFit(X,Y)
    breaks = my_pwlf.fit(n_segments).round(2)
    y_breaks = my_pwlf.predict(breaks).round(2)
    maxConstraint = X[Y.round(2) == mu].max()
    minConstraint = X[Y.round(2) == mu].min()
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

def CoSimulation():
    text = open("in\in_main.idf").read()
    NextFile = open("in\in.IDF","wt")
    NextFile.write(text[:166851] + "\n" + textGenerator() + text[173349:])
    NextFile.close()
    os.system("RunEPlus.bat in in")

def simulation():
    for timestep in range(1,24):
        model = gp.Model("optim")
        z = model.addVar(name="z") # value of the objective function
        x = model.addVar(name="x") # next temperature
        delta_u = model.addVar(name="delta_u",lb=-20, ub=+20)
        # setpoint of the building
        # Adding constraints
        model.addConstr(x == 0.0076 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep] 
                        - 0.063 * X["Previous Temperature"][timestep-1] + delta_u * 0.4392 + 1.2504 + 
                        X["Previous Temperature"][timestep-1])
    
        my_pwlf = approx_PWLF(updated_mean[0],updated_mean[1],updated_mean[2],updated_mean[3])
        
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
        model.addConstr(z == (0.106 * X["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"][timestep]
                              - 0.0625 * x 
                              - 0.3430 * (X["setpoint"][timestep-1] + delta_u) + 7.6881) # * Electricity_Price[timestep]
                              - eta * (my_pwlf[1][0] * x1 + my_pwlf[1][1] * x2 + my_pwlf[1][2] * x3 + 
                                       my_pwlf[1][3] * x4 + my_pwlf[1][4] * x5)
                              + 10000)
        
        model.setObjective(z, GRB.MINIMIZE)
        model.optimize()
        
        X["setpoint"][timestep + 1] = k_p * round(model.getVars()[2].x,2) + X["setpoint"][timestep]
        X["alphas set"][timestep] = np.array(dist_alphas.sample())
        CoSimulation()
        X["Previous Temperature"] = ReadExcel()['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)']
        X["Energy"] = ReadExcel()['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) ']
        
#%% initialization
mu = 0.50
eta = 10
k_p = 0.1
n_c = 3
ro = 0.5

# Different initializations for variables for the comfort profiles
# These variables are 'a', 'b', 'loc', 'scale'

#%% running model
# for counter in range(5):
    X = pd.read_csv("SF_Main_24.csv")
    X = X[48:]
    X = X.reset_index()
    X["Hours"] = X.index
    X["Previous Temperature"] = None
    X["Previous Temperature"][0] = 23    
    X["setpoint"] = 29.44
    X["setpoint"][0:2] = 20
    
    CoSimulation()
    X["Previous Temperature"] = ReadExcel()['LIVING_UNIT1:Zone Mean Air Temperature [C](TimeStep)']
    X["Energy"] = ReadExcel()['CENTRAL SYSTEM_UNIT1:Air System Total Cooling Energy [J](TimeStep) ']
    
    X["alphas set"] = None

    # Calling the environment and saving the iteration
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    simulation()
    X.to_csv("X_round_" + str(counter) + ".csv")
    #%%
    X["Comfort cost"] = beta.pdf(X['Previous Temperature'], Ind_best['beta']['a'], Ind_best['beta']['b'],Ind_best['beta']['loc'],
               Ind_best['beta']['scale'])/Y2.max()


#%%
plt.plot(np.linspace(18,30,1200), beta.pdf(np.linspace(18,30,1200), 
         Generic_best['beta']['a'], Generic_best['beta']['b'],
         Generic_best['beta']['loc'],Generic_best['beta']['scale'])/Y1.max())
plt.plot(np.linspace(18,30,1200), beta.pdf(np.linspace(18,30,1200), 
         Ind_best['beta']['a'], Ind_best['beta']['b'],
         Ind_best['beta']['loc'],Ind_best['beta']['scale'])/Y2.max())
# Newfit = beta.pdf(np.linspace(18,30,1200), updated_mean[0],updated_mean[1],updated_mean[2],updated_mean[3])
plt.legend(["Generic","Final","updated"])

#%%
# temp = X['Previous Temperature'] < np.quantile(X['Previous Temperature'],0.5)
plt.plot(X['Previous Temperature'],X['Final cost'],'o')
plt.xlabel("temperature")
plt.ylabel("Probability")
#%%
X["Count"] = X["Comfort cost"].round(2) * 100
Empty_comfort = np.repeat(X['Previous Temperature'],X["Count"])
f = Fitter(Empty_comfort,distributions=["beta"])                          
f.fit()
# f.summary()
Comfort_best = f.get_best(method = 'sumsquare_error')

plt.plot(np.linspace(18,30,1200), beta.pdf(np.linspace(18,30,1200), 
         Generic_best['beta']['a'], Generic_best['beta']['b'],
         Generic_best['beta']['loc'],Generic_best['beta']['scale'])/Y1.max())
plt.plot(np.linspace(18,30,1200), beta.pdf(np.linspace(18,30,1200), 
         Ind_best['beta']['a'], Ind_best['beta']['b'],
         Ind_best['beta']['loc'],Ind_best['beta']['scale'])/Y2.max())
Newfit = beta.pdf(np.linspace(18,30,1200), Comfort_best['beta']['a'], Comfort_best['beta']['b'],
         Comfort_best['beta']['loc'],Comfort_best['beta']['scale'])
plt.plot(np.linspace(18,30,1200), Newfit/Newfit.max())
# Newfit = beta.pdf(np.linspace(18,30,1200), updated_mean[0],updated_mean[1],updated_mean[2],updated_mean[3])
plt.legend(["Generic","Final","updated"])
plt.savefig("new.png", dpi = 2000)

