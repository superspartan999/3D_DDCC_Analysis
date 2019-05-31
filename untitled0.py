# -*- coding: utf-8 -*-
"""
Created on Fri May 31 00:31:57 2019

@author: Kun
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


directory = 'E:\\10nmAlGaN\\Bias -42'
directory = 'E:\Google Drive\Research\Guillaume'
os.chdir(directory)

filename = "graph"
G=read_json_file(directory+"\\"+ filename + ".json")

df=pd.read_csv("LED4In-out.vg_0.00.vd_3.20.vs_0.00.unified", delimiter=',')

node_map=df[['x','y','z']].copy()
#round up values in node map to prevent floating point errors
rounded_nodes=node_map.round(decimals=10)

sorted_data=df.round({'x':10,'y':10,'z':10})
sorted_data=sorted_data.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)

if sorted_data['Ec'].min()<0:   
    sorted_data['Ec']=sorted_data['Ec']-sorted_data['Ec'].min()
    

#create dataframes for each xyz dimension in the mesh. this creates a dimension list 
#that gives us the total no. of grid points in any given direction
unique_x=rounded_nodes['x'].unique()
unique_y=rounded_nodes['y'].unique()
unique_z=rounded_nodes['z'].unique()


#sort these dataframes in acsending order
xvalues=pd.DataFrame(unique_x).sort_values([0],ascending=True).reset_index(drop=True)
yvalues=pd.DataFrame(unique_y).sort_values([0],ascending=True).reset_index(drop=True)
zvalues=pd.DataFrame(unique_z).sort_values([0],ascending=True).reset_index(drop=True)


start=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == 0)]

end=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == zvalues.iloc[len(zvalues)-1][0])]
s=nx.shortest_path_length(G,start.index.values[0],end.index.values[0],weight='weight')

h=nx.shortest_path(G,start.index.values[0],end.index.values[0],weight='weight')

path=pd.DataFrame(index=range(len(h)),columns={'Node','x','y','z'})

for i,val in enumerate(h):
    path.iloc[i]=sorted_data.iloc[val][['Node','x','y','z']]
   
    
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')


x=path['x'].values

y=path['y'].values

z=path['z'].values

ax.set_xlim(0, xvalues[0].iat[-1]) 
ax.set_ylim(0,yvalues[0].iat[-1])
ax.set_zlim(0,zvalues[0].iat[-1])
ax.scatter(x, y, z, c='r', marker='o')

