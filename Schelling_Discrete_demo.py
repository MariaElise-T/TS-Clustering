#!/usr/bin/env python
# coding: utf-8

# In[80]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import statistics
from random import uniform, seed
from math import sqrt
from dataclasses import dataclass


# In[81]:


@dataclass
class Agent:
    AgentID: int
    Color: str 
    Happy: int 
    x_loc: float
    y_loc: float
    neighbors: []


# In[82]:


def findNeighborDistances(Agent_Array):
    for i in range(0, len(Agent_Array)):
        Agent_Array[i].neighbors.clear()
        for j in range(0, len(Agent_Array)):
            Agent_Array[i].neighbors.append((Agent_Array[j].AgentID, sqrt((Agent_Array[i].x_loc-Agent_Array[j].x_loc)**2+(Agent_Array[i].y_loc-Agent_Array[j].y_loc)**2)))


# In[83]:


def assignHappiness(Agent_Array, bias, k, plots):
    homogeneity_vec = []
    for i in range(0, len(Agent_Array)):
        closestNeighbors = sorted(Agent_Array[i].neighbors, key = lambda x: x[1])[1:(k+1)]
        matchedNeighbors = 0
        AgentColor = Agent_Array[i].Color
        for j in range(0, len(closestNeighbors)):
            neighborID = closestNeighbors[j][0]
            for Agent in Agent_Array:
                if Agent.AgentID == neighborID:
                    if Agent.Color==AgentColor:
                        matchedNeighbors = matchedNeighbors + 1
        homogeneity_vec.append(matchedNeighbors/k)
        if(matchedNeighbors/k >= bias):
            Agent_Array[i].Happy = 1
        else:
            Agent_Array[i].Happy = 0
    if(plots == True):
        print("Mean homogeneity: ", statistics.mean(homogeneity_vec), "% \n")
    return statistics.mean(homogeneity_vec)


# In[84]:


def decideWhoMoves(Agent_Array, Agent_Grid):
    N = len(Agent_Array)
    for i in range(0, N):
        if(Agent_Array[i].Happy == 0):
            x_rand = np.random.choice(np.arange(40), 1)[0]
            y_rand = np.random.choice(np.arange(40), 1)[0]
            while(Agent_Grid[x_rand, y_rand] != -1):
                x_rand = np.random.choice(np.arange(40), 1)[0]
                y_rand = np.random.choice(np.arange(40), 1)[0]
            Agent_Array[i].x_loc=x_rand
            Agent_Array[i].y_loc=y_rand


# In[85]:


def plotNeighborhood(Agent_Array):
    df = pd.DataFrame( columns=['x_loc', 'y_loc', 'color'])
    i=0
    for Agent in Agent_Array:
        df.loc[i] = [Agent.x_loc, Agent.y_loc, Agent.Color]
        i = i+1
    plt.scatter(df["x_loc"], df["y_loc"], c = df["color"])
    plt.show()


# In[86]:


def checkHappiness(Agent_Array, plots):
    NumHappy = 0
    for Agent in Agent_Array:
        if Agent.Happy == 1:
            NumHappy = NumHappy+1
    if(plots == True):
        print("Percent of happy agents: ", 100*NumHappy/len(Agent_Array), "% \n")
    return 100*NumHappy/len(Agent_Array)


# In[87]:


def runSim(Agent_Array, Agent_Grid, steps, bias, k, plots):
    if(plots == True):
        print("Iteration 0: \n")
        plotNeighborhood(Agent_Array)
    percHappyVec = []
    meanHomogeneousVec = []
    for i in range(0, steps):
        findNeighborDistances(Agent_Array)
        meanHomogeneity = assignHappiness(Agent_Array, bias, k, plots)
        meanHomogeneousVec.append(meanHomogeneity)
        decideWhoMoves(Agent_Array, Agent_Grid)
        if(plots == True):
            print("Iteration ", i+1, " : \n")
        percHappy = checkHappiness(Agent_Array, plots)
        percHappyVec.append(percHappy)
        if(plots == True):
            plotNeighborhood(Agent_Array)
            #print("Percent happy at iteration ", i, ": ", percHappy, "\n")
    return percHappyVec, meanHomogeneousVec


# In[88]:


def runSimFullData(Agent_Array, Agent_Grid, steps, bias, k, plots):
    if(plots == True):
        print("Iteration 0: \n")
        plotNeighborhood(Agent_Array)
    fullSimVec = []
    for i in range(0, steps):
        findNeighborDistances(Agent_Array)
        assignHappiness(Agent_Array, bias, k)
        decideWhoMoves(Agent_Array, Agent_Grid)
        if(plots == True):
            print("Iteration ", i+1, " : \n")
        df = pd.DataFrame( columns=['x_loc', 'y_loc', 'color'])
        i=0
        for Agent in Agent_Array:
            df.loc[i] = [Agent.x_loc, Agent.y_loc, Agent.Color]
            i = i+1
        df_green = df[df["color"] == "green"]
        df_orange = df[df["color"] == "orange"]
        df_matrix_green = np.zeros((40, 40))
        df_matrix_orange = np.zeros((40, 40))
        for index, row in df_green.iterrows():
            df_matrix_green[row['x_loc'], row['y_loc']] = 1
        for index, row in df_orange.iterrows():
            df_matrix_orange[row['x_loc'], row['y_loc']] = 1
        vec_green = df_matrix_green.flatten()
        vec_orange = df_matrix_orange.flatten()
        vec_one_hot = np.concatenate((vec_green, vec_orange), axis=None)
        fullSimVec = np.concatenate((fullSimVec, vec_one_hot), axis=None)
        if(plots == True):
            plotNeighborhood(Agent_Array)
    return fullSimVec


# In[89]:


def setupAndRunSim(steps, bias, k, orangeNum, greenNum, plots):
    Agent_Array = []
    Agent_Grid = np.ones((40,40))
    Agent_Grid = -1*Agent_Grid
    for i in range(0, orangeNum):
        x_rand = np.random.choice(np.arange(40), 1)[0]
        y_rand = np.random.choice(np.arange(40), 1)[0]
        while(Agent_Grid[x_rand, y_rand] != -1):
            x_rand = np.random.choice(np.arange(40), 1)[0]
            y_rand = np.random.choice(np.arange(40), 1)[0]
        Agent_Grid[x_rand][y_rand] = i
        Agent_Array.append(Agent(AgentID=i, Color="orange", Happy=0, x_loc=x_rand, y_loc=y_rand, neighbors=[]))
    for i in range(0, greenNum):
        x_rand = np.random.choice(np.arange(40), 1)[0]
        y_rand = np.random.choice(np.arange(40), 1)[0]
        while(Agent_Grid[x_rand, y_rand] != -1):
            x_rand = np.random.choice(np.arange(40), 1)[0]
            y_rand = np.random.choice(np.arange(40), 1)[0]
        Agent_Grid[x_rand, y_rand] = i
        Agent_Array.append(Agent(AgentID=i+250, Color="green", Happy=0, x_loc=x_rand, y_loc=y_rand, neighbors=[]))
    finalPercHappyVec, finalPercHomoVec = runSim(Agent_Array, Agent_Grid, steps, bias, k, plots)
    return finalPercHappyVec, finalPercHomoVec


# In[90]:


def setupAndRunSimFullOutput(steps, bias, k, orangeNum, greenNum, plots):
    Agent_Array = []
    Agent_Grid = np.ones((40,40))
    Agent_Grid = -1*Agent_Grid
    for i in range(0, orangeNum):
        x_rand = np.random.choice(np.arange(40), 1)[0]
        y_rand = np.random.choice(np.arange(40), 1)[0]
        while(Agent_Grid[x_rand, y_rand] != -1):
            x_rand = np.random.choice(np.arange(40), 1)[0]
            y_rand = np.random.choice(np.arange(40), 1)[0]
        Agent_Grid[x_rand][y_rand] = i
        Agent_Array.append(Agent(AgentID=i, Color="orange", Happy=0, x_loc=x_rand, y_loc=y_rand, neighbors=[]))
    for i in range(0, greenNum):
        x_rand = np.random.choice(np.arange(40), 1)[0]
        y_rand = np.random.choice(np.arange(40), 1)[0]
        while(Agent_Grid[x_rand, y_rand] != -1):
            x_rand = np.random.choice(np.arange(40), 1)[0]
            y_rand = np.random.choice(np.arange(40), 1)[0]
        Agent_Grid[x_rand, y_rand] = i
        Agent_Array.append(Agent(AgentID=i+250, Color="green", Happy=0, x_loc=x_rand, y_loc=y_rand, neighbors=[]))
    fullSimVec = runSimFullData(Agent_Array, Agent_Grid, steps, bias, k, plots)
    return fullSimVec


from multiprocessing import Pool

def generateSims(j):
    tolerance = uniform(0, 1)
    vec_hap, vec_homo = setupAndRunSim(10, tolerance, 10, 250, 250, False)
    df_inputs = pd.DataFrame([vec_hap])
    df_inputs2 = pd.DataFrame([vec_homo])
    df_outputs = pd.DataFrame([tolerance])
    for i in range(99):
        tolerance = uniform(0, 1)
        vec_hap, vec_homo = setupAndRunSim(10, tolerance, 10, 250, 250, False)
        df_inputs.loc[len(df_inputs.index)] =  vec_hap
        df_inputs2.loc[len(df_inputs2.index)] =  vec_homo
        df_outputs.loc[len(df_outputs.index)] =  tolerance
    input_file_name = "train_inputs_hap_" + str(j) + ".csv"
    input2_file_name = "train_inputs_homo_" + str(j) + ".csv"
    output_file_name = "train_outputs_hap_" + str(j) + ".csv"
    df_inputs.to_csv(input_file_name, encoding='utf-8', index=False)
    df_inputs2.to_csv(input2_file_name, encoding='utf-8', index=False)
    df_outputs.to_csv(output_file_name, encoding='utf-8', index=False)
    
def run_generate_sims(operation, input, pool):
    pool.map(operation, input)
    
if __name__ == '__main__':
    processes_count = 20
    processes_pool = Pool(processes_count)
    run_generate_sims(generateSims, range(100), processes_pool)   






