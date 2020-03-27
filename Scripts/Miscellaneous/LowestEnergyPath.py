# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 23:44:16 2019

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
from matplotlib import cm
from itertools import *
import heapq
from scipy.spatial import KDTree

directory = 'E:\\10nmAlGaN\\Bias -42'
directory = 'E:\\Google Drive\\Research\\Guillaume'
directory = 'C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume'

directory = 'D:\\3D Simulations\\32nmAlGaN017\\Bias0'

file = 'p_structure_0.17_32nm-out.vg_0.00.vd_0.00.vs_0.00.unified'
os.chdir(directory)


df=pd.read_csv(file, delimiter=',')

sorted_data,xvalues,yvalues,zvalues=processdf(df)

zslice=extract_slice(sorted_data,'z',zvalues.iloc[len(zvalues)-1][0]/2,drop=True)

zmap=zslice[['x','y','Ec']].reset_index().round({'x':10,'y':10,'z':10})



def edgeweight2d(source,target,space,merged):
    
    average=(merged[source][2]+merged[target][2])
    print(average/space)

    
    return average



def coordtonode2d(x_idx,y_idx,unique_x,unique_y):
    
    max_y=len(unique_y)

    
    index = x_idx * max_y + y_idx 
    return index





x=zmap['x'].values

y=zmap['y'].values

z=zmap['Ec'].values

x_vals, x_idx = np.unique(x, return_inverse=True)
y_vals, y_idx = np.unique(y, return_inverse=True)

Ec_array = np.empty(x_vals.shape + y_vals.shape)

Ec_array.fill(np.nan)

Ec_array[x_idx, y_idx] = zmap['Ec'].values

merged=np.vstack((x,y,z))

merged=np.transpose(merged)

dictm=dict(enumerate(merged))

G=nx.Graph()

space= np.diff(x_vals)[0]
G.add_nodes_from(dictm.keys())
for key, n in list(G.nodes.items()):

    n['pos']=dictm[key][0:2].tolist()
    n['pot']=dictm[key][2]


xy=zmap[['index','x','y']]
point=xy[['x','y']].values
point_tree=KDTree(point)


for key, n in list(G.nodes.items()):
    
    neighbourhood=point_tree.query_ball_point(point[key], 6.05e-8)
    
    
    neighbourhood.remove(key)
    for neigh in neighbourhood:
        G.add_edge(key,neigh,weight=edgeweight2d(key,neigh,space,merged))

source=1
target=2599
def k_shortest_paths(G, source, target, k, weight=None):
     return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def mypath(G,source,target):

    pathlist=[]
    place_holder=99999999999999
    unseenNodes=list(G.nodes)
    energy={node: place_holder for node in unseenNodes}
    iteration=[place_holder for node in unseenNodes]

    energy[source]=G.node[source]['pot']
    i=1

    while unseenNodes:
        current_node = min(unseenNodes, key=lambda node: energy[node])

        neighbourhood=list(G.neighbors(current_node))
        for neighbour in neighbourhood:
             energy[neighbour]=G.node[neighbour]['pot']
        
        iteration[current_node]=i
        i=i+1

        unseenNodes.remove(current_node)
        
    while target != source:
       backtrackneigh=list(G.neighbors(target))
       neighdf=pd.DataFrame(columns={'neighbour','iteration'})
       neighdf['neighbour']=backtrackneigh
       for neigh in backtrackneigh:
           neighdf.loc[neighdf['neighbour']==neigh,'iteration']=iteration[neigh]

           
       step=neighdf.loc[neighdf['iteration'].idxmin()]['neighbour']
       pathlist.append(step)
       print(step)
       target=step
       

       
    return pathlist
           


    