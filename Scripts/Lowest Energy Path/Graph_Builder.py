# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:50:26 2019

@author: Clayton
"""

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
import heapq
import pickle

length=20

material='AlGaN'
comp=1
directory='D:/'+str(length)+'nm'+material+''+str(comp)
directory='D:\\Guillaume Data\\LEDIndiumCompo_'+str(comp)+'Al_'+str(length)+'Angs_\\Bias3'
##directory = 'D:\\3D Simulations\\8nmAlN\\Bias0'
#if material=='AlGaN':
#    material='p_structure'
os.chdir(directory)
#file= 'p_structure_0.17_'+str(length)+'nm-out.vg_0.00.vd_0.00.vs_0.00.unified'
#file='AlGaN_'+str(comp)+'_'+str(length)+'nm_-out.vg_0.00.vd_0.00.vs_0.00.unified'
#directory = 'C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume\\'
#file= 'LED4In-out.vg_0.00.vd_3.20.vs_0.00.unified'

file = 'LEDIndiumCompo_'+str(comp)+'Al_'+str(length)+'Angs_-out.vg_0.00.vd_0.00.vs_0.00.unified'

file = 'p_structure_0.17_32nm-out.vg_0.00.vd_0.00.vs_0.00.unified'
no_of_paths=10


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
sorted_data=sorted_data[(sorted_data['z']>4.0e-6) & (sorted_data['z']<(4.0e-6+thickness))].reset_index(drop=True)
#sorted_data=sorted_data[(sorted_data['z']>4e-6) & (sorted_data['z']<4.1e-6)].reset_index(drop=True)


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

#Ecdf=sorted_data[['x','y','z','Ec','Ev']].copy()
#Ecdf=Ecdf.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
#Ecdf['Bgap']=Ecdf['Ec']-Ecdf['Ev']
#Bgapdf=Ecdf[['x','y','z','Bgap']].copy()
#Bgaparr=Bgapdf.values
Evdf=sorted_data[['x','y','z','Ev']].copy()
Evdf['Ev']=Evdf['Ev'].abs()
Evarr=Evdf.values
#
#Jpdf=sorted_data[['x','y','z','Hole_current']].copy()


dictEc=dict(enumerate(Evarr))
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


save_json('C:\\Users\\Clayton\\Desktop\\Guillaume\\'+str(length)+'nm'+material+''+str(comp)+'_graph.js',G)