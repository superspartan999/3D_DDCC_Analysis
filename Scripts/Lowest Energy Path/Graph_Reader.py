# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:42:04 2020

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
import random
import statistics



G=read_json_file('C:\\Users\\Clayton\\Desktop\\Guillaume\\'+str(length)+'nm'+material+''+str(comp)+'_graph.js')
pathlist = pickle.load( open( 'C:\\Users\\Clayton\\Desktop\\Guillaume\\'+str(length)+'nm'+material+''+str(comp)+'pathlist.txt', "rb" ) )

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')    

for h in pathlist[0:100]:

    path=pd.DataFrame(index=range(len(h)),columns={'Node','x','y','z'})
    
    for i,val in enumerate(h):
        path.iloc[i]=sorted_data.iloc[val][['Node','x','y','z']]
    
    path['x']=path['x'].astype(float)
    path['y']=path['y'].astype(float)
    path['z']=path['z'].astype(float)

    
    x=path['x'].values
    
    y=path['y'].values
    
    z=path['z'].values
    
#    ax.set_xlim(0, xvalues[0].iat[-1]) 
#    ax.set_ylim(0,yvalues[0].iat[-1])
#    ax.set_zlim(0,zvalues[0].iat[-1])
    ax.scatter(x, y, z)