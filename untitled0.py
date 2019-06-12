# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 02:16:34 2019

@author: Clayton
"""
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

def mypath(G,source,target):

    pathlist=[]

    unseenNodes=list(G.nodes).copy()
    energy={node: 999999999 for node in unseenNodes}
    iteration={node: None for node in unseenNodes}

    energy[source]=G.node[source]['pot']
    i=1
    while unseenNodes:
        current_node = min(unseenNodes, key=lambda node: energy[node])
        
        if energy[current_node] == 999999999:
                break
        neighbourhood=list(G.neighbors(current_node))
        for neighbour in neighbourhood:
             print(neighbour)
             energy[neighbour]=G.node[neighbour]['pot']
        
        iteration[current_node]=i
        i=i+1
        print(i)
        print(current_node)
        unseenNodes.remove(current_node)
        
    while target != source:
       backtrackneigh=list(G.neighbors(target))
       neighdf=pd.DataFrame(columns={'neighbour','iteration'})
       neighdf['neighbour']=backtrackneigh
       for neigh in backtrackneigh:
           neighdf.loc[neighdf['neighbour']==neigh,'iteration']=iteration[neigh]
           print(iteration[neigh])
           
       step=neighdf.loc[neighdf['iteration'].idxmin()]['neighbour']
       pathlist.append(step)
       target=step
    return pathlist
           

G=read_json_file('C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume\\graph.json')
