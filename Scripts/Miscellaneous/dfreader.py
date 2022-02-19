# -*- coding: utf-8 -*-
"""
Created on Fri May 31 01:33:08 2019

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
           

def mypath2(G,source,target):       
    pathlist=[]
    place_holder=99999999
    unseenNodes=list(G.nodes)
    energy=[place_holder for node in unseenNodes]
    iteration=[place_holder for node in unseenNodes]
    heapq.heapify(energy)
    heapq.heapify(iteration)

    energy[source]=G.node[source]['pot']
    i=1

    while unseenNodes:
        current_node = heapq.nsmallest(1,unseenNodes, key=lambda node: energy[node])[0]
        
        if energy[current_node] == 999999999:
                break
        neighbourhood=list(G.neighbors(current_node))
        for neighbour in neighbourhood:
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
           
       neighdf=neighdf[neighdf['iteration']!=place_holder]
       step=neighdf.loc[neighdf['iteration'].idxmin()]['neighbour']
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
       print(step)
       pathlist.append(step)
       target=step
    return pathlist

            
            
            
source=57
target=2000
h=mypath(G,source,target)

#shortestpaths=[]
#for path in k_shortest_paths(G, 1, 2600, 3, weight='weight'):
#    shortestpaths.append(shortestpaths)

#
#h=mypath3(G,source,target)     


path=pd.DataFrame(index=range(len(h)),columns={'x','y'})
for i,val in enumerate(h):
        path.loc[i]=zmap.iloc[val][['x','y']]

nodeweights=0

#path=pd.DataFrame(index=range(len(h)),columns={'x','y'})
#for i,val in enumerate(h):
#        path.loc[i]=zmap.iloc[val][['x','y']]

#nodeweights=0
##
#for node in h:
#    nodeweights=G.node[node]['pot']+nodeweights
##    
#averagenodeenergy=nodeweights/len(h)


#
#
#

#
#
#xx,yy=np.meshgrid(x_vals,y_vals)
#zz=np.zeros_like(xx)
#
#for xind, x in enumerate(x_vals):
#    for yind, y in enumerate(y_vals):
#        zz[xind][yind]=zmap['Ec'].iloc[coordtonode2d(xind,yind, x_vals,y_vals)] 
#
#fig = plt.figure()
#CS=plt.contourf(x_vals,y_vals,Ec_array,30,cmap=cm.plasma) 
#
#CS2=plt.contour(x_vals,y_vals,Ec_array, colors='black',linewidths=0.5)
#
#plt.scatter(path['x'],path['y'], s=1)
#plt.clabel(CS2)




xx,yy=np.meshgrid(x_vals,y_vals)
zz=np.zeros_like(xx)

for xind, x in enumerate(x_vals):
    for yind, y in enumerate(y_vals):
        zz[xind][yind]=zmap['Ec'].iloc[coordtonode2d(xind,yind, x_vals,y_vals)] 

fig = plt.figure()
CS=plt.contourf(x_vals,y_vals,Ec_array,30,cmap=cm.jet) 

CS2=plt.contour(x_vals,y_vals,Ec_array, colors='black',linewidths=0.5)

plt.scatter(path['x'],path['y'], s=1)
plt.clabel(CS2)


fig = plt.figure()
CS=plt.contourf(x_vals,y_vals,Ec_array,30,cmap=cm.plasma) 

CS2=plt.contour(x_vals,y_vals,Ec_array, colors='black',linewidths=0.5)

plt.scatter(path['x'],path['y'], s=20)
plt.clabel(CS2)




#fig = plt.figure()

#
##fig = plt.figure()

#ax = fig.add_subplot(111, projection='3d')  
#ax.plot_surface(xx,yy,zz,cmap=cm.plasma,alpha=0.5) 
#ax.scatter(path['x'],path['y'],0.58,s=50,c='b') 



#    
#    Xp=merged[key][0]+space
#    Xn=merged[key][0]-space
#    Yp=merged[key][1]+space
#    Yn=merged[key][1]-space
    
#    if Xp > x_vals[len(x_vals)-1]:
#        Xp=merged[key][0]
#        
#    if Xn < 0:
#        Xn=merged[key][0]    
#    
#    if Yp > y_vals[len(y_vals)-1]:
#        Yp=merged[key][0]
#        
#    if Xn < 0:
#        Yn=merged[key][0]      


#    
#    print('---')
#    print(Xneighp)
#    print(Xneighn)
#    print(Yneighp)
#    print(Yneighn)    
#    if len(Xneighp)==0:
#        g=0
#    
#    else:
#        counter+=1
#        G.add_edge(key,Xneighp[0],weight=edgeweight2d(key,Xneighp[0],space,merged))
#        
#
#    if len(Xneighn)==0:
#        g=0
#    
#    else:
#        counter+=1
#        G.add_edge(key,Xneighn[0],weight=edgeweight2d(key,Xneighn[0],space,merged))        
#        
#    if len(Yneighp)==0:
#        g=0
#    
#    else:
#        counter+=1
#        G.add_edge(key,Yneighp[0],weight=edgeweight2d(key,Yneighp[0],space,merged))
#        
#
#    if len(Yneighn)==0:
#        g=0
#    
#    else:
#        counter+=1
#        G.add_edge(key,Yneighn[0],weight=edgeweight2d(key,Yneighn[0],space,merged))                
    

        
#def neighbours(node,G,zmap):
#    neighbourhood=list(G.neighbors(node))
#    print(neighbourhood)
#    coords=np.empty(shape=(len(neighbourhood)+1,2))
#    
#    for key,n in enumerate(neighbourhood):
#        coords[key]=np.transpose(zmap.iloc[n][['x','y']].values)
#    
#
#    coords[len(neighbourhood)]=zmap.iloc[node][['x','y']]
#    h=pd.DataFrame(coords,columns=['x','y'])
#    return h
    

    

    
    
    