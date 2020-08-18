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
        
def mypath3(G,source,target):
    pathlist=[]
    place_holder=99999999.0
    energy = [place_holder] * len(list(G.nodes))
    iteration2=[place_holder for node in list(G.nodes)]
    queue = [(G.node[source]['pot'], source)]
    i=1
    while queue:
        current_energy, current_node = heapq.heappop(queue)
        iteration2[current_node]=i
        i+=1
        if energy[current_node] is place_holder: # v is unvisited
            energy[current_node] = G.node[current_node][u'pot']
        neighbourhood= list(G.neighbors(current_node))
        for neigh in neighbourhood:
                if energy[neigh] is place_holder:
                    energy[neigh]=G.node[neigh][u'pot']
                    heapq.heappush(queue, (G.node[neigh][u'pot'],neigh))

    while target != source:
       backtrackneigh=list(G.neighbors(target))
       neighdf=pd.DataFrame(columns={'neighbour','iteration'})
       neighdf['neighbour']=backtrackneigh
       for neigh in backtrackneigh:
           if neigh not in pathlist:
               neighdf.loc[neighdf['neighbour']==neigh,'iteration']=iteration2[neigh]
               
       neighdf=neighdf.dropna().reset_index(drop=True)
       neighdf=neighdf[neighdf['iteration']!=place_holder]
       step=neighdf.loc[(neighdf['iteration']==min(neighdf['iteration']))]['neighbour'].values[0]
#       print(step)
       pathlist.append(step)
       target=step
    return pathlist

def mypath3(G,source,target):
    pathlist=[]
    place_holder=99999999.0
    energy = [place_holder] * len(list(G.nodes))
    iteration2=[place_holder for node in list(G.nodes)]
    queue = [(G.node[source]['pot'], source)]
    i=1
    while queue:
        current_energy, current_node = heapq.heappop(queue)
        iteration2[current_node]=i
        i+=1
        if energy[current_node] is place_holder: # v is unvisited
            energy[current_node] = G.node[current_node]['pot']
        neighbourhood= list(G.neighbors(current_node))
        for neigh in neighbourhood:
                if energy[neigh] is place_holder:
                    energy[neigh]=G.node[neigh]['pot']
                    heapq.heappush(queue, (G.node[neigh]['pot'],neigh))

    while target != source:
       backtrackneigh=list(G.neighbors(target))
       neighdf=pd.DataFrame(columns={'neighbour','iteration'})
       neighdf['neighbour']=backtrackneigh
       for neigh in backtrackneigh:
           if neigh not in pathlist:
               neighdf.loc[neighdf['neighbour']==neigh,'iteration']=iteration2[neigh]
               
       neighdf=neighdf.dropna().reset_index(drop=True)
       neighdf=neighdf[neighdf['iteration']!=place_holder]
       step=neighdf.loc[(neighdf['iteration']==min(neighdf['iteration']))]['neighbour'].values[0]
#       print(step)
       pathlist.append(step)
       target=step
    return pathlist


length=8
#
directory='C:\\Users\\Clayton\\Desktop\\'+str(length)+'nmAlGaN'
directory = 'D:\\3D Simulations\\8nmAlN\\Bias0'
file= 'p_structure_0.17_'+str(length)+'nm-out.vg_0.00.vd_0.00.vs_0.00.unified'
#directory = 'C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume\\'
#file= 'LED4In-out.vg_0.00.vd_3.20.vs_0.00.unified'
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

       
#source=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])& \
#                      (sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z']==zvalues.iloc[0][0])]
#
#target=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])& \
#                      (sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z']==zvalues.iloc[246][0])]      
#source=source.index[0]
#target=target.index[0]
#h=mypath3(G,source,target)


thickness=(length)*1e-7
    
#
#start=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y']== yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == 0)]
#    
#end=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == zvalues.iloc[len(zvalues)-1][0])]            
##start=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y']== yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == 4E-6)]
#h=mypath3(G,start.index[0],end.index[0])
#
#nodeweights=0
##
#for node in h:
#    nodeweights=G.node[node]['pot']+nodeweights
##    
#averagenodeenergy=nodeweights/len(h)
#
#path=pd.DataFrame(index=range(len(h)),columns={'Node','x','y','z'})
#
#for i,val in enumerate(h):
#    path.iloc[i]=sorted_data.iloc[val][['Node','x','y','z']]
##
#fig = plt.figure()
#
#ax = fig.add_subplot(111, projection='3d')
#
#path['x']=path['x'].astype(float)
#path['y']=path['y'].astype(float)
#path['z']=path['z'].astype(float)
#
#
#x=path['x'].values
#
#y=path['y'].values
#
#z=path['z'].values
#
#
##ax.set_xlim(0, xvalues[0].iat[-1]) 
##ax.set_ylim(0,yvalues[0].iat[-1])
##ax.set_zlim(4e-6,zvalues[0].iat[-1])
#ax.scatter(x, y, z)


#


minindex = abs(zvalues[0]).idxmin()
maxindex=  abs(zvalues[0]-(zvalues.iloc[minindex][0]+thickness)).idxmin()
 
start_slice=extract_slice(sorted_data,'z',zvalues.iloc[minindex][0])
end_slice=extract_slice(sorted_data,'z',zvalues.iloc[len(zvalues)-11][0])

#start_slice=extract_slice(sorted_data,'z',zvalues.iloc[0][0])
#end_slice=extract_slice(sorted_data,'z',zvalues.iloc[len(zvalues)-1][0])
 
start_node_list=np.array(start_slice.index)
end_node_list=np.array(end_slice.index)

pathlist=[]
for i in range(1,no_of_paths):
    random_start=random.choice(start_node_list)
    random_target=random.choice(end_slice.index)
    pathway=mypath3(G,random_start,random_target)
    pathlist.append(pathway)
    print(i)
    
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
#    
save_json('C:\\Users\\Clayton\\Desktop\\Guillaume\\'+str(length)+'nmAlGaNgraph.js',G)
energylist=[]
for h in pathlist[0:no_of_paths]: 
    nodeweights=0
    #
    for node in h:
        nodeweights=G.node[node]['pot']+nodeweights
#    
    averagenodeenergy=nodeweights/len(h)
    energylist.append(averagenodeenergy)
    
with open('C:\\Users\\Clayton\\Desktop\\Guillaume\\'+str(length)+'nmpathlist.txt','wb') as fp:
    pickle.dump(pathlist,fp)