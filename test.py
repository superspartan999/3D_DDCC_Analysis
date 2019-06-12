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
from functions import *


def mypath(G,source,target):       
    pathlist=[]

    unseenNodes=list(G.nodes).copy()
    energy=[999999999 for node in unseenNodes]
    iteration=[99999999999999 for node in unseenNodes]
    heapq.heapify(energy)
    heapq.heapify(iteration)

    energy[source]=G.node[source]['pot']
    i=1

    while target in unseenNodes:
        current_node = heapq.nsmallest(1,unseenNodes, key=lambda node: energy[node])[0]
        
        if energy[current_node] == 999999999:
                break
        neighbourhood=list(G.neighbors(current_node))
        for neighbour in neighbourhood:
             energy[neighbour]=G.node[neighbour]['pot']
        
        iteration[current_node]=i
        i=i+1
#        print(i)
#        print(current_node)
        unseenNodes.remove(current_node)
        
    while target != source:
       backtrackneigh=list(G.neighbors(target))
       neighdf=pd.DataFrame(columns={'neighbour','iteration'})
       neighdf['neighbour']=backtrackneigh
       for neigh in backtrackneigh:
           neighdf.loc[neighdf['neighbour']==neigh,'iteration']=iteration[neigh]
#           print(iteration[neigh])
           
       neighdf=neighdf[neighdf['iteration']!='place_holder']
       step=neighdf.loc[neighdf['iteration'].idxmin()]['neighbour']
       pathlist.append(step)
       target=step
    return pathlist
           
def mypath2(G,source,target):       
    pathlist=[]


    unseenNodes=list(G.nodes).copy()
    energy=[9999.0 for node in unseenNodes]
    iteration=[99999999999999.0 for node in unseenNodes]
    testdf=pd.DataFrame({'Energy':energy,'Iteration':iteration})
    testdf.iloc[source]['Energy']=G.node[source]['pot']
    i=1

    while target in unseenNodes:
        current_node = testdf['Energy'].iloc[unseenNodes].idxmin()
        testdf['Energy'].idxmin()
        if testdf['Energy'].iloc[current_node] == 999999999:
                break
        neighbourhood=list(G.neighbors(current_node))
        for neighbour in neighbourhood:
             energy[neighbour]=G.node[neighbour]['pot']
        
        testdf['Iteration'].iloc[current_node]=i
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
           
       neighdf=neighdf[neighdf['iteration']!='place_holder']
       step=neighdf.loc[neighdf['iteration'].idxmin()]['neighbour']
       pathlist.append(step)
       target=step
    return pathlist
G=read_json_file('C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume\\graph.json')
directory = 'C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume\\'
file= 'LED4In-out.vg_0.00.vd_3.20.vs_0.00.unified'
h=mypath(G,25,25991)

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

start=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == 3.0657e-06)]

end=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == 6.9902e-06)]

h=mypath(G,start.index,25991)
##function to plot band diagram
#def band_diagram_z(df1):
#    #find all the values of z and put them in a list
#    zvalues = df1['z'].unique()
#    cols={}
#    #create dataframe for conduction band and valence band
#    Ecvalues=pd.DataFrame(columns=['z','Ec'])
#    Evvalues=pd.DataFrame(columns=['z','Ev'])
#    i=0
#    #loop through different z values along the device
#    for z in zvalues:
#        #extract x-y plane for a z value
#        zslice=extract_slice(df1,'z',z, drop=True)
#        
#        #average
#        averagezsliceEc=zslice['Ec'].mean()
#        averagezsliceEv=zslice['Ev'].mean()
#        d1={'z':z,'Ec':averagezsliceEc}
#        d2={'z':z,'Ev':averagezsliceEv}
#        Ecvalues.loc[i]=d1
#        Evvalues.loc[i]=d2
#        i=i+1
#
#
#
#    return Ecvalues,Evvalues
#
#Ecslices=band_diagram_z(sorted_data)