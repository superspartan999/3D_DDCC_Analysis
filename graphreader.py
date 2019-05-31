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


directory = 'E:\\10nmAlGaN\\Bias -42'
file= 'p_structure_0.17_10nm-out.vg_0.00.vd_-4.20.vs_0.00.unified'


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

Ecdf=sorted_data[['x','y','z','Ec']].copy()
Ecdf=Ecdf.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
Ecarr=Ecdf.values
dictEc=dict(enumerate(Ecarr, 1))
G=nx.Graph()
G.add_nodes_from(dictEc.keys())
for key, n in G.nodes.items():
    n['pos']=dictEc[key][0:3].tolist()
    n['pot']=dictEc[key][3]
    
        
for key, n in list(G.nodes.items())[:-1]:
    xneighs=NNX(key,xvalues,yvalues,zvalues)
    yneighs=NNY(key,xvalues,yvalues,zvalues)
    zneighs=NNZ(key,xvalues,yvalues,zvalues)
    
    if key==xneighs[0]:
       g=0
    else:
        G.add_edge(key,xneighs[0],weight=float(edgeweight(key,xneighs[0],xvalues,yvalues,zvalues,Ecdf)))
    
    if key==xneighs[1]:
       g=0
    else:

        G.add_edge(key,xneighs[1],weight=float(edgeweight(key,xneighs[1],xvalues,yvalues,zvalues,Ecdf)))


    if key==yneighs[0]:
       g=0
    else:

        G.add_edge(key,yneighs[0],weight=float(edgeweight(key,yneighs[0],xvalues,yvalues,zvalues,Ecdf)))
    
    if key==yneighs[1]:
        g=0
    else:
        G.add_edge(key,yneighs[1],weight=float(edgeweight(key,yneighs[1],xvalues,yvalues,zvalues,Ecdf)))
        
    if key==zneighs[0]:
       g=0
    else:
        G.add_edge(key,zneighs[0],weight=float(edgeweight(key,zneighs[0],xvalues,yvalues,zvalues,Ecdf)))
   
    if key==zneighs[1]:
      g=0
    else:
        G.add_edge(key,zneighs[1],weight=float(edgeweight(key,zneighs[1],xvalues,yvalues,zvalues,Ecdf)))
    if key%100000==0:
        print key
        