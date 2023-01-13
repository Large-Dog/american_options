#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2022)

# Initialise global parameters
s0 = 10       # initial stock price
mu = 0.05     # drift
sigma = 0.2   # volatility
T = 1         # maturity time 
r = 0.02      # risk-free rate
N = 5000      # number of time steps CHANGE TO 5000, 15 IS TEST VALUE
K = 10        # Strike Price

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
value, decision = makeMurricanTree(priceTree, time, dt, r, q)


#%% 3a)

# i)
# Find decision boundary using the decision tree to get a list the first occurrence
# where actual price > hold
boundary = []
for i in range(N+1):
    col = np.where(decision[:,i] == 1.0)[0]
    # print(i, col)
    if col.any():
       boundary.append((time[i], priceTree[:,i][col[0]]))
boundary = np.array(boundary) 

# Plotting Decision Boundary

fig, axes = plt.subplots()
axes.plot(boundary[:,0], boundary[:,1], color = "red")
axes.set_title("Exercise Boundary of American Put Option", fontsize = 20)
axes.set_xlabel("Time (t)", fontsize = 12)
axes.set_ylabel("Optimal Early Exercise Value", fontsize = 12)


# In[72]:


european = np.zeros([N+1,N+1])
european[:,N] = np.maximum(np.zeros(N+1), K-priceTree[:,N])
for time in range(N-1 , -1, -1):
        # print("TIME " + str(time))
        for price in range(0, time + 1):
            european[price][time] = np.exp(-r * dt) * (q * european[price, time+1] + (1-q) * european[price+1, time+1])
european


# In[195]:


#%%

# ii)
# Hedging Strategy at time = 0
S0 = np.linspace(1, 20, num=20)
alpha_0 = np.zeros(20)
beta_0 = np.zeros(20)

alpha_0_EU = np.zeros(20)
beta_0_EU = np.zeros(20)

for i in range(20):
    value0, decision0 = makeMurricanTree(makeTree(S0[i], N+1, up, down), time, dt, r, q)
    alpha_0[i] = (value0[0,1]-value0[1,1])/(S0[i]*up - S0[i]*down)
    beta_0[i] = (value0[0,1]-alpha_0[i]*S0[i]*up)/np.exp(r*dt)
    alpha_0_EU[i] = (european[0,1]-european[1,1])/(S0[i]*up - S0[i]*down)
    beta_0_EU[i] = (european[0,1]-alpha_0_EU[i]*S0[i]*up)/np.exp(r*dt)


# Plot
plt.plot(S0, alpha_0, 'r', label='Num. of Assets (American Put)')
plt.plot(S0, beta_0, 'y', label='Num. of Bank Account (American Put)')
plt.plot(S0, alpha_0_EU, 'blue', label='Num. of Assets (European Put)',linestyle= 'dashed')
plt.plot(S0, beta_0_EU, 'y', label='Num. of Bank Account(European Put)',linestyle= 'dashed')
plt.xlim(0,25)
plt.xlabel('Varying Asset Price at time 0')
plt.ylabel('Numbers of holdings')
plt.title('Figure 2: Hedging strategy of American vs European Put option')
plt.legend()
plt.show()


# In[273]:


plt.plot(S0, alpha_0, 'r', label='Num. of Assets (American Put)')
plt.plot(S0, beta_0, 'y', label='Num. of Bank Account (American Put)')
plt.xlim(0,25)
plt.xlabel('Varying Asset Price at time 0')
plt.ylabel('Numbers of holdings')
plt.title('Figure 2: Hedging strategy of American vs European Put option')
plt.legend()
plt.show()


# In[187]:


alpha0 = (value[0,1]-value[1,1])/(10*up - 10*down)
beta0 = (value[0,1]-alpha0*10*up)/np.exp(r*dt)


# In[208]:


# Hedging Strategy at time = 1/4
time_point = 0.25
alpha_1_4 = np.zeros(1250)
beta_1_4 = np.zeros(1250)

alpha_1_4_EU = np.zeros(1250)
beta_1_4_EU = np.zeros(1250)

for i in range(1250):
    alpha_1_4[i] = (value[i,1+1250]-value[i+1,1+1250])/(priceTree[i,1+1250] - priceTree[i+1,1+1250])
    beta_1_4[i] = (value[i,1+1250]-alpha_1_4[i]*priceTree[i,1+1250])/np.exp(r*(0.25+dt))
    
    alpha_1_4_EU[i] = (european[i,1+1250]-european[i+1,1+1250])/(priceTree[i,1+1250] - priceTree[i+1,1+1250])
    beta_1_4_EU[i] = (european[i,1+1250]-alpha_1_4[i]*priceTree[i,1+1250])/np.exp(r*(0.25+dt))

# Plot
plt.plot(priceTree[0:1250, 1250], alpha_1_4, 'r', label='Assets (American Put)')
plt.plot(priceTree[0:1250, 1250], beta_1_4, 'y', label='Bank Account (American Put)')
plt.plot(priceTree[0:1250, 1250], alpha_1_4, 'b', label='Assets (European Put)',linestyle= 'dashed')
plt.plot(priceTree[0:1250, 1250], beta_1_4, 'g', label='Bank Account (European Put)',linestyle= 'dashed')
plt.xlim(0,25)
plt.xlabel('Varying Asset Price at time 0.25')
plt.ylabel('Numbers of holdings')
plt.title('Figure 3: Hedging strategy of American vs European Put option at time 0.25')
plt.legend()
plt.show()



# In[252]:


# Hedging Strategy at time = 1/2
time_point = int(0.5*5000)
alpha_2_4 = np.zeros(2500)
beta_2_4 = np.zeros(2500)

alpha_2_4_EU = np.zeros(2500)
beta_2_4_EU = np.zeros(2500)

for i in range(2500):
    alpha_2_4[i] = (value[i,1+2500]-value[i+1,1+2500])/(priceTree[i,1+2500] - priceTree[i+1,1+2500])
    beta_2_4[i] = (value[i,1+2500]-alpha_2_4[i]*priceTree[i,1+2500])/np.exp(r*(dt))
    
    
    alpha_2_4_EU[i] = (european[i,1+2500]-european[i+1,1+2500])/(priceTree[i,1+2500] - priceTree[i+1,1+2500])
    beta_2_4_EU[i] = (european[i,1+2500]-alpha_2_4[i]*priceTree[i,1+2500])/np.exp(r*(dt))

# Plot
plt.plot(priceTree[0:2500, 2500], alpha_2_4, 'r', label='Assets (American Put)')
plt.plot(priceTree[0:2500, 2500], beta_2_4, 'y', label='Bank Account (American Put)')
plt.plot(priceTree[0:2500, 2500], alpha_2_4, 'b', label='Assets (European Put)',linestyle= 'dashed')
plt.plot(priceTree[0:2500, 2500], beta_2_4, 'g', label='Bank Account (European Put)',linestyle= 'dashed')
plt.xlim(0,25)
plt.xlabel('Varying Asset Price at time 0.5')
plt.ylabel('Numbers of holdings')
plt.title('Figure 3: Hedging strategy of American vs European Put option at time 0.5')
plt.legend()
plt.show()


# In[253]:


# Hedging Strategy at time = 3/4
time_steps = 3750
alpha_3_4 = np.zeros(3750)
beta_3_4 = np.zeros(3750)

alpha_3_4_EU = np.zeros(3750)
beta_3_4_EU = np.zeros(3750)

print(alpha_3_4)
print(beta_3_4)

for i in range(3750):
    alpha_3_4[i] = (value[i,1+3750]-value[i+1,1+3750])/(priceTree[i,1+3750] - priceTree[i+1,1+3750])
    beta_3_4[i] = (value[i,1+3750]-alpha_3_4[i]*priceTree[i,1+3750])/np.exp(r*(0.75+dt))
    
    alpha_3_4_EU[i] = (european[i,1+3750]-european[i+1,1+3750])/(priceTree[i,1+3750] - priceTree[i+1,1+3750])
    beta_3_4_EU[i] = (european[i,1+3750]-alpha_3_4[i]*priceTree[i,1+3750])/np.exp(r*(0.75+dt))

# Plot
plt.plot(priceTree[0:3750, 3750], alpha_3_4, 'r', label='Num. of Assets')
plt.plot(priceTree[0:3750, 3750], beta_3_4, 'y', label='Num. of Bank Account')
plt.plot(priceTree[0:3750, 3750], alpha_3_4, 'b', label='Assets (European Put)',linestyle= 'dashed')
plt.plot(priceTree[0:3750, 3750], beta_3_4, 'g', label='Bank Account (European Put)',linestyle= 'dashed')

plt.xlim(0,25)
plt.xlabel('Varying Asset Price at time 0.75')
plt.ylabel('Numbers of holdings')
plt.title('Figure 3: Hedging strategy of American vs European Put option at time 0.75')
plt.legend()
plt.show()


# In[286]:


# Hedging Strategy at Maturity time T = 1
time_steps = 3750
alpha_4_4 = np.zeros(5000)
beta_4_4 = np.zeros(5000)

alpha_4_4_EU = np.zeros(5000)
beta_4_4_EU = np.zeros(5000)

print(alpha_4_4)
print(beta_4_4)

for i in range(5000):
    if priceTree[i,5000] < K:
        alpha_4_4[i] = -1
        beta_4_4[i] = 10
    else:
        alpha_4_4[i] = 0
        beta_4_4[i] = 0

# Plot
plt.plot(S0, alpha_0, 'b',label='t=0')
plt.plot([10], [alpha0], marker="o", markersize=10, markeredgecolor="purple",
markerfacecolor="purple")
plt.plot(priceTree[0:5000:, 5000], alpha_4_4, color="brown", label='t=1')
plt.xlim(0,25)
plt.xlabel('Underlying Asset Price at time t')
plt.ylabel('Numbers of risky assets holding')
plt.title('Figure 4: Hedging strategy of American Put option')
plt.legend()


# In[289]:


plt.plot(S0, alpha_0, 'b',label='t=0')
plt.plot([10], [alpha0], marker="o", markersize=10, markeredgecolor="purple",
markerfacecolor="purple")
plt.plot(priceTree[0:1250, 1250], alpha_1_4, 'r', label='t=0.25')
plt.plot(priceTree[0:2500, 2500], alpha_2_4, 'y', label='t=0.5')
plt.plot(priceTree[0:3750, 3750], alpha_3_4, 'g', label='t=0.75')
plt.plot(priceTree[0:5000, 5000], alpha_4_4, color="brown", label='t=1')
plt.xlim(0,25)
plt.xlabel('Underlying Asset Price at time t')
plt.ylabel('Numbers of risky assets holding')
plt.title('Figure 4: Hedging strategy of American Put option')
plt.legend()


# In[288]:


plt.plot(S0, beta_0, 'b',label='t=0')
plt.plot([10], [beta0], marker="o", markersize=10, markeredgecolor="purple",
markerfacecolor="purple")
plt.plot(priceTree[0:1250, 1250], beta_1_4, 'r', label='t=0.25')
plt.plot(priceTree[0:2500, 2500], beta_2_4, 'y', label='t=0.5')
plt.plot(priceTree[0:3750, 3750], beta_3_4, 'g', label='t=0.75')
plt.plot(priceTree[0:5000, 5000], beta_4_4, color="brown", label='t=1')
plt.xlim(0,25)
plt.xlabel('Underlying Asset Price at time t')
plt.ylabel('Numbers of risk-free assets holding')
plt.title('Figure 5: Hedging strategy of American Put option')
plt.legend()


# In[3]:


#%% Functions for price tree, value tree, and decision tree for american puts

def makeTree(s0, steps, up, down):
    tree = np.zeros((steps, steps))
    tree[0] = s0
    
    for i in range(1,steps):
        tree[0][i] = tree[0][i-1] * up
        for j in range(1, steps):
            tree[j][i] = tree[j-1, i-1] * down
            
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

def makeMurricanTree(pT, t, dt, r, q):
    # calculates option value and strategy given price tree and payoff function
    
    # make value tree, European and decision tree frame
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
    
    # calculate the optimal strategy between the hold and intrinsic
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


# In[ ]:


# Changing Volatility and risk-free rates
s0 = 10       # initial stock price
mu = 0.05     # drift
#sigma = 0.1   # volatility is varing now
T = 1         # maturity time 
rs = 0.02      # risk-free rate

N2 = 2500

time2 = np.linspace(0, T, N2+1)
dt2 = T/N2

# Here, the decision boundary do not vary with mu so we do not have to consider it
# sigmama variation:

colours = ["red", "blue", "green", "brown", "yellow", "purple", "black"]
fig, axes = plt.subplots()
labs = []
j = 0

for sigma in [0.05, 0.1, 0.2, 0.3, 0.5]:
    up = np.exp(rs * dt2 + sigma * (dt2 ** (1/2)))
    down = np.exp(rs * dt2 - sigma * (dt2 ** (1/2)))
    priceTree = makeTree(s0, N2 + 1, up, down)
    q = (1 - np.exp(-sigma * dt2**(1/2))) / (np.exp(sigma * dt2**(1/2)) - np.exp(-sigma * dt2**(1/2)))
    
    value, decision = makeMurricanTree(priceTree, time2, dt2, rs, q)
    

    alpha_1_4 = np.zeros(1250)
    beta_1_4 = np.zeros(1250)

    for i in range(1250):
        alpha_1_4[i] = (value[i,1+1250]-value[i+1,1+1250])/(priceTree[i,1+1250] - priceTree[i+1,1+1250])
        beta_1_4[i] = (value[i,1+1250]-alpha_1_4[i]*priceTree[i,1+1250])/np.exp(r*(0.25+dt))
    
    plt.plot(priceTree[0:1250, 1250], alpha_1_4, color = colours[j], label=("σ = " + str(sigma) + ""))
    plt.plot(priceTree[0:1250, 1250], beta_1_4, color = colours[j], label=("Bank Account σ = " + str(sigma) + ""))
    plt.xlim(0,20)
    plt.xlabel('Underlying Asset Price at time t = 0.25')
    plt.ylabel('Numbers of risk-free assets')
    #plt.title('Figure 8: Hedging strategy of r = 2%')
    plt.legend()

    
    j += 1

