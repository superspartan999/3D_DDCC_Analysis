# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:16:36 2019

@author: Clayton
"""


from functions import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from math import floor, sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import os
import networkx as nx
from networkx.readwrite import json_graph
import simplejson as json
import heapq
import pickle
import statistics


lengtharray=[1,2,4,8,16,32,64]
comparray=[1,2,4,8,16,32,64]
#lengtharray=[10,30,50]
energyarray=np.empty(len(lengtharray))
max_E=np.empty((len(lengtharray)))
averageenergy=np.empty((len(lengtharray)))
spread=np.empty(len(lengtharray))
median=np.empty(len(lengtharray))
maxbandgap=np.empty(len(lengtharray))
bandgappath=np.empty(len(lengtharray))
meanbandgap=np.empty(len(lengtharray))
minbandgap=np.empty(len(lengtharray))
comparray= [16]
lengtharray= [32]

comp
#for iteration,comp in enumerate(comparray):
for iteration,length in enumerate(lengtharray):
    directory='D:/Bandgap Path and Network/'
    ##directory = 'D:\\3D Simulations\\8nmAlN\\Bias0'
#    file= 'AlGaN_'+str(comp)+'_'+str(length)+'nm_-out.vg_0.00.vd_0.00.vs_0.00.unified'
#    file= 'D:/'+str(length)+'nmAlGaN'+str(comp)+'/AlGaN_'+str(comp)+'_'+str(length)+'nm_-out.vg_0.00.vd_0.00.vs_0.00.unified'
    file='D:/'+str(length)+'nmAlGaN/p_structure_0.17_'+str(length)+'nm-out.vg_0.00.vd_0.00.vs_0.00.unified'
    #directory = 'C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume\\'
    #file= 'LED4In-out.vg_0.00.vd_3.20.vs_0.00.unified'
    
    
    
    os.chdir(directory)
    df=pd.read_csv(file, delimiter=',')
    
    node_map=df[['x','y','z']].copy()
    #round up values in node map to prevent floating point errors
    rounded_nodes=node_map.round(decimals=10)
    #
    ##sort the nodes in ascending order
    sorted_nodes=rounded_nodes.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
    sorted_data=df.round({'x':10,'y':10,'z':10})
    sorted_data=sorted_data.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
     
    thickness=1e-7*length
#    sorted_data=sorted_data[(sorted_data['z']>4.0e-6) & (sorted_data['z']<(4.0e-6+thickness))].reset_index(drop=True)
##    sorted_data=sorted_data[(sorted_data['z']>4e-6) & (sorted_data['z']<4.1e-6)].reset_index(drop=True)
#    
##    
#    if sorted_data['Ec'].min()<0:   
#        sorted_data['Ec']=sorted_data['Ec']-sorted_data['Ec'].min()
        
#    G=read_json_file(str(length)+'nmAlGaN'+str(comp)+'graph.js')
    G=read_json_file(str(length)+'nmAlGaNgraph.js')

    #G=read_json_file('C:\\Users\\Clayton\\Desktop\\Guillaume\\'+str(length)+'nmAlGaNgraph.js')
    with open(str(length)+'nmAlGaNpathlist.txt', "rb") as fp:
#    with open(str(length)+'nmAlGaN'+''+str(comp)+'pathlist.txt', "rb") as fp:   # Unpicklin
        pathlist = pickle.load(fp)
        
    energylist=[]
#    for h in pathlist[0:1000]: 
#        nodeweights=0
#        #
#        for node in h:
#            nodeweights=G.node[node]['pot']+nodeweights
#    #     
#        averagenodeenergy=nodeweights/len(h)
#        energylist.append(averagenodeenergy)
    
    bandgaplist=[]
    for h in pathlist[0:1000]: 
        nodeweights=0
        #
        for node in h:
            nodeweights=(sorted_data.iloc[node]['Ec']-sorted_data.iloc[node]['Ev'])+nodeweights
    #     
        averagenodeenergy=nodeweights/len(h)
        bandgaplist.append(averagenodeenergy)
        
    valencelist=[]
    for h in pathlist[0:1000]: 
        nodeweights=0
        #
        for node in h:
            nodeweights=(sorted_data.iloc[node]['Ev'])+nodeweights
    #     
        averagenodeenergy=nodeweights/len(h)
        energylist.append(averagenodeenergy)
        
    
    
    
#    plt.hist(energylist, bins=arange(min(energylist), max(energylist) + 0.001, 0.005),label=str(length))

    energyarray[iteration]=statistics.mean(energylist)
    bandgappath[iteration]=statistics.mean(bandgaplist)
    spread[iteration]=np.std(energylist)
    median[iteration]=statistics.median(energylist)
    max_E[iteration]=sorted_data['Ec'].max()
    averageenergy[iteration]=sorted_data['Ec'].mean()
    bgap=sorted_data['Bgap']=sorted_data['Ec']-sorted_data['Ev']
    maxbandgap[iteration]=bgap.max()
    meanbandgap[iteration]=bgap.mean()
    minbandgap[iteration]=bgap.min()
    EcEv=band_diagram_z(sorted_data)
    path_Ec=np.empty(len(pathlist[0]))
    path_Ev=np.empty(len(pathlist[0]))
#    
#    for i,h in enumerate(pathlist[0]):
#
#        path_Ec[i]=sorted_data.iloc[h]['Ec']
#        path_Ev[i]=sorted_data.iloc[h]['Ev']
#        
        
    plt.plot(EcEv[0]['z']/1e-7,EcEv[0]['Ec'], label=str(length))
    plt.plot(EcEv[1]['z']/1e-7,EcEv[1]['Ev'],label=str(length))
#    #
#    
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')    

for h in pathlist[0:1]:

    path=pd.DataFrame(index=range(len(h)),columns={'Node','x','y','z'})
    
    for i,val in enumerate(h):
        path.iloc[i]=sorted_data.iloc[val][['Node','x','y','z']]
    
    path['x']=path['x'].astype(float)
    path['y']=path['y'].astype(float)
    path['z']=path['z'].astype(float)

    
    x=path['x'].values/1e-7
    
    y=path['y'].values/1e-7
    
    z=path['z'].values/1e-7
    
#    ax.set_xlim(0, xvalues[0].iat[-1]) 
#    ax.set_ylim(0,yvalues[0].iat[-1])
#    ax.set_zlim(0,zvalues[0].iat[-1])
    ax.scatter(x, y, z)