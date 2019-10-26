# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:48:37 2019

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

def mypath3(G,source,target):
    pathlist=[]
    place_holder=99999999
    A = [None] * len(list(G.nodes))
    iteration=[place_holder for node in list(G.nodes)]
    queue = [(G.node[source]['pot'], source)]
    i=1
    while queue:
        current_energy, current_node = heapq.heappop(queue)
        iteration[current_node]=i
        i+=1
        if A[current_node] is None: # v is unvisited
            A[current_node] = G.node[current_node][u'pot']
            for neigh in list(G.neighbors(current_node)):
                if A[neigh] is None:
                    heapq.heappush(queue, (G.node[neigh][u'pot'],neigh))

    while target != source:
       backtrackneigh=list(G.neighbors(target))
       neighdf=pd.DataFrame(columns={'neighbour','iteration'})
       neighdf['neighbour']=backtrackneigh
       for neigh in backtrackneigh:
           neighdf.loc[neighdf['neighbour']==neigh,'iteration']=iteration[neigh]


       step=neighdf.loc[neighdf['iteration'].idxmin()]['neighbour']
       print(step)
       pathlist.append(step)
       target=step
    return pathlist

directory = 'E:\\10nmAlGaN\\Bias -42'
file= 'p_structure_0.17_10nm-out.vg_0.00.vd_-4.20.vs_0.00.unified'

directory = 'C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume\\'
file= 'LED4In-out.vg_0.00.vd_3.20.vs_0.00.unified'



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
sorted_data=sorted_data[(sorted_data['z']>3e-6) & (sorted_data['z']<7e-6)]

if sorted_data['Ec'].min()<0:   
    sorted_data['Ec']=sorted_data['Ec']-sorted_data['Ec'].min()
    

#create dataframes for each xyz dimension in the mesh. this creates a dimension list 
#that gives us the total no. of grid points in any given direction
unique_x=sorted_data['x'].unique()
unique_y=sorted_data['y'].unique()
unique_z=sorted_data['z'].unique()


#sort these dataframes in acsending order
xvalues=pd.DataFrame(unique_x).sort_values([0],ascending=True).reset_index(drop=True)
yvalues=pd.DataFrame(unique_y).sort_values([0],ascending=True).reset_index(drop=True)
zvalues=pd.DataFrame(unique_z).sort_values([0],ascending=True).reset_index(drop=True)

Ecdf=sorted_data[['x','y','z','Ec']].copy()
Ecdf=Ecdf.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
Ecarr=Ecdf.values
dictEc=dict(enumerate(Ecarr))
G=nx.Graph()
G.add_nodes_from(dictEc.keys())
for key, n in G.nodes.items():
    n['pos']=dictEc[key][0:3].tolist()
    n['pot']=dictEc[key][3]
    
        
for key, n in list(G.nodes.items()):
    xneighs=NNX(key,xvalues,yvalues,zvalues)
    yneighs=NNY(key,xvalues,yvalues,zvalues)
    zneighs=NNZ(key,xvalues,yvalues,zvalues)
    
    if key==xneighs[0]:
       g=0
    else:
        G.add_edge(key,xneighs[0])
    
    if key==xneighs[1]:
       g=0
    else:

        G.add_edge(key,xneighs[1])


    if key==yneighs[0]:
       g=0
    else:

        G.add_edge(key,yneighs[0])
    
    if key==yneighs[1]:
        g=0
    else:
        G.add_edge(key,yneighs[1])
        
    if key==zneighs[0]:
       g=0
    else:
        G.add_edge(key,zneighs[0])
   
    if key==zneighs[1]:
      g=0
    else:
        G.add_edge(key,zneighs[1])
    if key%100000==0:
        print(key)
        
h=mypath3(G,1,13090)