    # -*- coding: utf-8 -*-
"""
Created on Sat Oct 8 21:39:52 2022

@author: micha
"""
### IMPORTANT: PLEASE RUN FUNCTIONS AT BOTTOM BEFORE RUNNING EVERYTHING ELSE
#              AS EVERYTHING DEPENDS ON THOSE FUNCTIONS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(2022)

# Initialise global parameters
s0 = 10       # initial stock price
mu = 0.05     # driff
sigma = 0.2   # volatility
T = 1         # maturity 
r = 0.02      # risk-free rate
N = 5000      # number of time steps CHANGE TO 5000, 15 IS TEST VALUE
K = 10        # Strike

time = np.linspace(0, T, N+1)
dt = T/N

# Calculate up and down factors
up = np.exp(r * dt + sigma * (dt ** (1/2)))
down = np.exp(r * dt - sigma * (dt ** (1/2)))

# Initializing S tree with N time steps
priceTree = makeTree(s0, N + 1, up, down)

# calculating q probability
q = (1 - np.exp(-sigma * dt**(1/2))) / (np.exp(sigma * dt**(1/2)) - np.exp(-sigma * dt**(1/2)))

#%% constructing american option tree and decision tree associated with option tree
value, decision = makeMurricanTree(priceTree, time, dt, r, q, N)

#%% 3a) i)

# Find decision boundary using the decision tree to get a list the first occurrence
# where actual price > hold
boundary = []
for i in range(N+1):
    col = np.where(decision[:,i] == 1.0)[0]
    # print(i, col)
    if col.any():
       boundary.append((time[i], priceTree[:,i][col[0]], value[:,i][col[0]]))
boundary = np.array(boundary) 

# Plotting Decision Boundary

fig, axes = plt.subplots()
axes.plot(boundary[:,0], boundary[:,1], color = "red")
axes.set_title("Exercise Boundary of American Put Option", fontsize = 20)
axes.set_xlabel("t", fontsize = 12)
axes.set_ylabel("Optimal Early Exercise Value", fontsize = 12)

#%% 3a) ii) hedging, refer to other code file. I was not in charge for coding


#%% 3a) iii) sigma variation

N2 = 5000

time2 = np.linspace(0, T, N2+1)
dt2 = T/N2

# Here, the decision boundary do not vary with mu so we do not have to consider it
# sigma variation:

colours = ["red", "blue", "green", "brown", "yellow", "purple", "black"]
fig, axes = plt.subplots()
labs = []
j = 0

for sigs in [0.05, 0.1, 0.2, 0.3, 0.5]:
    up = np.exp(r * dt2 + sigs * (dt2 ** (1/2)))
    down = np.exp(r * dt2 - sigs * (dt2 ** (1/2)))
    priceTree = makeTree(s0, N2 + 1, up, down)
    q = (1 - np.exp(-sigs * dt2**(1/2))) / (np.exp(sigs * dt2**(1/2)) - np.exp(-sigs * dt2**(1/2)))
    
    value, decision = makeMurricanTree(priceTree, time2, dt2, r, q, N2)
    
    boundary = []
    for i in range(N2+1):
        col = np.where(decision[:,i] == 1.0)[0]
        # print(i, col)
        if col.any():
           boundary.append((time[i], priceTree[:,i][col[0]]))
    boundary = np.array(boundary) 

    # Plotting Decision Boundary
    
    print("Calculation finished fpr sig = " + str(sigs))

    axes.plot(boundary[:,0], boundary[:,1], color = colours[j])
    j += 1
    labs.append(r"$\sigma = " + str(sigs) + "$")

axes.legend(labs)
axes.set_title("Exercise Boundary of American Put Option Various Sigma", fontsize = 20)
axes.set_xlabel("t", fontsize = 12)
axes.set_ylabel("Optimal Early Exercise Value", fontsize = 12)



#%% 3a) iii) risk free variation

N2 = 5000

time2 = np.linspace(0, T, N2+1)
dt2 = T/N2

# Here, the decision boundary do not vary with mu so we do not have to consider it
# sigmama variation:

colours = ["red", "blue", "green", "brown", "yellow", "purple", "black"]
fig, axes = plt.subplots()
labs = []
j = 0

for rs in [0.005, 0.01, 0.02, 0.03, 0.05]:
    up = np.exp(rs * dt2 + sigma * (dt2 ** (1/2)))
    down = np.exp(rs * dt2 - sigma * (dt2 ** (1/2)))
    priceTree = makeTree(s0, N2 + 1, up, down)
    q = (1 - np.exp(-sigma * dt2**(1/2))) / (np.exp(sigma * dt2**(1/2)) - np.exp(-sigma * dt2**(1/2)))
    
    value, decision = makeMurricanTree(priceTree, time2, dt2, rs, q, N2)
    
    boundary = []
    for i in range(N2+1):
        col = np.where(decision[:,i] == 1.0)[0]
        # print(i, col)
        if col.any():
           boundary.append((time[i], priceTree[:,i][col[0]]))
    boundary = np.array(boundary) 

    # Plotting Decision Boundary
    
    print("Calculation finished for r = " + str(rs))

    axes.plot(boundary[:,0], boundary[:,1], color = colours[j])
    j += 1
    labs.append("r = " + str(rs) + "")



axes.legend(labs)
axes.set_title("Exercise Boundary of American Put Option Various Risk Free Rate", fontsize = 20)
axes.set_xlabel("t", fontsize = 12)
axes.set_ylabel("Optimal Early Exercise Value", fontsize = 12)



#%% 3b) simulating path and taking its kde
Nsims = 10000

sim = assetSim(s0, r, mu, sigma, dt, time, Nsims)
densities = intersection(sim, value, time,  boundary, dt, Nsims)

densities = pd.DataFrame(densities)
PnL = densities[2] - value[0,0]


not_exercised = np.full((Nsims - PnL.shape[0]), -value[0, 0])

PnL = np.concatenate((PnL, not_exercised))
    

#%% 3b) plotting

fig, axes = plt.subplots(1,2)

for i, row in enumerate(axes):
    if i == 1:
        df = pd.DataFrame(PnL)
        row.hist(PnL, density = True, bins = 10, color = "green", alpha = 0.5)
        sns.kdeplot(data=df, x=0, color = "black")
        row.set_xlabel("Profit", fontsize = 12)
        row.legend(["Estimated KDE's"])

    if i == 0:
        df = pd.DataFrame(densities)
        row.hist(densities[0], density = True, bins = 10, alpha = 0.5)
        sns.kdeplot(data=df, x=0, color = "black", ax = row)
        row.set_xlabel("Time of Exercise", fontsize = 12)


    
plt.suptitle(r"Profits and Losses And Times of Exercise Under Base Assumptions", fontsize = 16)

#%% 3b) mu modulation

mus = [0.02, 0.05, 0.1]
Nsims = 10000
colours = ["red", "blue", "green", "brown", "yellow", "purple", "black"]
c = 0
fig, axes = plt.subplots(1,2)
labs = []

for mu in mus:
    sim = assetSim(s0, r, mu, sigma, dt, time, Nsims)
    densities = intersection(sim, value, time,  boundary, dt, Nsims)
    
    densities = pd.DataFrame(densities)
    PnL = densities[2] - value[0,0]


    not_exercised = np.full((Nsims - PnL.shape[0]), -value[0, 0])
    print("Calculation done for mu = " + str(mu))
    labs.append(r"$\mu = " + str(mu) + "$")
    PnL = np.concatenate((PnL, not_exercised))
    
    
    for i, row in enumerate(axes):
        if i == 1:
            df = pd.DataFrame(PnL)
            sns.kdeplot(data=df, x=0, color = colours[c])
            row.set_xlabel("Profit", fontsize = 12)

        if i == 0:
            df = pd.DataFrame(densities)
            sns.kdeplot(data=df, x=0, color = colours[c], ax = row)
            row.set_xlabel("Time of Exercise", fontsize = 12)
            
    c += 1
    
title = r"Profits and Losses And Times of Exercise Under $\mu$ Modulation"
plt.legend(labs)
plt.suptitle(title, fontsize = 16)

#%% 3b) sigma modulation


sigmas = [0.1, 0.15, 0.2, 0.25, 0.3]
Nsims = 10000
colours = ["red", "blue", "green", "brown", "yellow", "purple", "black"]
c = 0
fig, axes = plt.subplots(1,2)
labs = []

for sigma in sigmas:
    
    # Calculate up and down factors
    up = np.exp(r * dt + sigma * (dt ** (1/2)))
    down = np.exp(r * dt - sigma * (dt ** (1/2)))

    # Initializing S tree with N time steps
    priceTree = makeTree(s0, N + 1, up, down)

    # calculating q probability
    q = (1 - np.exp(-sigma * dt**(1/2))) / (np.exp(sigma * dt**(1/2)) - np.exp(-sigma * dt**(1/2)))
    
    value, decision = makeMurricanTree(priceTree, time, dt, r, q, N)
    
    boundary = []
    for i in range(N+1):
        col = np.where(decision[:,i] == 1.0)[0]
        # print(i, col)
        if col.any():
           boundary.append((time[i], priceTree[:,i][col[0]], value[:,i][col[0]]))
    boundary = np.array(boundary) 
    
    
    print("Calculation done for sigma = " + str(sigma))
    
    sim = assetSim(s0, r, mu, sigma, dt, time, Nsims)
    densities = intersection(sim, value, time, boundary, dt, Nsims)
    
    densities = pd.DataFrame(densities)
    PnL = densities[2] - value[0,0]


    not_exercised = np.full((Nsims - PnL.shape[0]), -value[0, 0])
    
    labs.append(r"$\sigma = " + str(sigma) + "$")
    PnL = np.concatenate((PnL, not_exercised))
    
    print("Simulation done for sigma = " + str(sigma))
    
    for i, row in enumerate(axes):
        if i == 1:
            df = pd.DataFrame(PnL)
            sns.kdeplot(data=df, x=0, color = colours[c])
            row.set_xlabel("Profit", fontsize = 12)

        if i == 0:
            df = pd.DataFrame(densities)
            sns.kdeplot(data=df, x=0, color = colours[c], ax = row)
            row.set_xlabel("Time of Exercise", fontsize = 12)
            
    c += 1
    
title = r"Profits and Losses And Times of Exercise Under $\sigma$ Modulation"
plt.legend(labs)
plt.suptitle(title, fontsize = 16)

#%%3b) Risk free Variation

rs = [0.005, 0.02, 0.05]
Nsims = 10000
colours = ["red", "blue", "green", "brown", "yellow", "purple", "black"]
c = 0
fig, axes = plt.subplots(1,2)
labs = []

for r in rs:
    
    # Calculate up and down factors
    up = np.exp(r * dt + sigma * (dt ** (1/2)))
    down = np.exp(r * dt - sigma * (dt ** (1/2)))

    # Initializing S tree with N time steps
    priceTree = makeTree(s0, N + 1, up, down)

    # calculating q probability
    q = (1 - np.exp(-sigma * dt**(1/2))) / (np.exp(sigma * dt**(1/2)) - np.exp(-sigma * dt**(1/2)))
    
    value, decision = makeMurricanTree(priceTree, time, dt, r, q, N)
    
    boundary = []
    for i in range(N+1):
        col = np.where(decision[:,i] == 1.0)[0]
        # print(i, col)
        if col.any():
           boundary.append((time[i], priceTree[:,i][col[0]], value[:,i][col[0]]))
    boundary = np.array(boundary) 
    
    
    print("Calculation done for r = " + str(r))
    
    sim = assetSim(s0, r, mu, sigma, dt, time, Nsims)
    densities = intersection(sim, value, time, boundary, dt, Nsims)
    
    densities = pd.DataFrame(densities)
    PnL = densities[2] - value[0,0]


    not_exercised = np.full((Nsims - PnL.shape[0]), -value[0, 0])
    
    labs.append("r = " + str(r))
    PnL = np.concatenate((PnL, not_exercised))
    
    print("Simulation done for r = " + str(r))
    
    for i, row in enumerate(axes):
        if i == 1:
            df = pd.DataFrame(PnL)
            sns.kdeplot(data=df, x=0, color = colours[c])
            row.set_xlabel("Profit", fontsize = 12)

        if i == 0:
            df = pd.DataFrame(densities)
            sns.kdeplot(data=df, x=0, color = colours[c], ax = row)
            row.set_xlabel("Time of Exercise", fontsize = 12)
            
    c += 1
    
title = r"Profits and Losses And Times of Exercise Under r Modulation"
plt.legend(labs)
plt.suptitle(title, fontsize = 16)


#%% Supporting functions

def makeTree(s0, steps, up, down):
    tree = np.zeros((steps, steps))
    tree[0] = s0
    
    for i in range(1,steps):
        tree[0][i] = tree[0][i-1] * up
        for j in range(1, steps):
            tree[j][i] = tree[j-1][i-1] * down
            
    return tree

def murrica(price):
    # calulates american put payoff
    if K - price < 0:
        return 0
    else:
        return K - price


def strat(price, comparison = 0):
    # returns 1 if price > comparison, 0 otherwise
    if price > comparison:
        out = 1
    else:
        out = 0
    return out

def makeMurricanTree(pT, t, dt, r, q, N):
    # calculates option value and strategy given price tree and payoff function
    
    # make value tree and decision tree frame
    dims = pT.shape
    vals = np.zeros(dims)
    decision = np.zeros(dims)
    boundary = []
    
    # start at end time 
    vals[:,-1] = np.array(list(map(murrica, pT[:,-1])))
    decision[:, -1] = np.array(list(map(strat, vals[:,-1])))
    
    
    temp = np.array(list(map(strat, vals[:,-1])))
    spotted = np.where(temp == 1)[0]
    boundary.append([N, spotted])
    
    # calculate the optimal strategy between the hold and european (intrinsic)
    # by propogating backwards starting from end time
    for time in range(N-1 , -1, -1):
        # print("TIME " + str(time))
        for price in range(0, time + 1):
            up = vals[price, time + 1]
            down = vals[price + 1, time + 1]
            hold = np.exp(-r * dt) * (q * up + (1-q) * down)
            intrinsic = murrica(pT[price][time])
            # print(time, price)
            # print(hold, intrinsic)
            optimal = max(hold, intrinsic)
            
            vals[price][time] = optimal
            decision[price][time] = strat(optimal, hold)

    return vals, decision

    
def truthToNums(x):
    if x:
        out = 1
    else:
        out = 0
    return out
    
def assetSim(s0, r, mu, sigma, dt, time, sims = 10000):
    sim = np.zeros((sims, time.shape[0]))
    sim[:, 0] = s0

    
    # calculate up probability p
    p = (1/2) * (1 + ((mu - r) - sigma**2 / 2) / sigma * dt**(1/2)) 
    
    for i in range(0, time.shape[0] - 1):
        # calculate up/down decision at each step
        a = np.random.uniform(size = sims)
        b = a < p
        decision = np.array(list(map(truthToNums, b)))
        direction = (decision * 2) - 1
        
        sim[:, i+1] = sim[:, i] * np.exp(r * dt + sigma * dt**(1/2) * direction)
        
        
    return sim

            
def intersection(sim, value, time, boundary, dt, nsims = 10_000):
    out = []
    for i in range(nsims):
        for bd in boundary:
            t_index = int(bd[0]/dt)
            if bd[1] >= sim[i, t_index]:
                out.append([bd[0], bd[1], bd[2]])
                break
    return out
            
            


