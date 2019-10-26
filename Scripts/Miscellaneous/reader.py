# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:27:40 2019

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


directory = 'E:\\10nmAlGaN\\Bias -42'
file= 'p_structure_0.17_10nm-out.vg_0.00.vd_-4.20.vs_0.00.unified'


directory= 'C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume'
#
##directory= 'C:\\Users\\Clayton\\Desktop\\50nmAlGaN\\Bias -42'
#
#directory = "/Users/claytonqwah/Documents/Google Drive/Research/Transport Structure Project/3D data/10nmAlGaN/Bias -42"
file= 'LED4In-out.vg_0.00.vd_3.20.vs_0.00.unified'


os.chdir(directory)
df=pd.read_csv(file, delimiter=',')


sorted_data, xvalues,yvalues,zvalues= processdf(df)

    
G=graphfromdf(df,xvalues,yvalues,zvalues)   


#start=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == 0)]
#
#end=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == zvalues.iloc[len(zvalues)-1][0])]
##s=nx.shortest_path_length(G,start.index.values[0],end.index.values[0],weight='weight')
##h=nx.shortest_path(G,start.index.values[0],end.index.values[0],weight='weight')
#
##
#averagenodeenergy=averageenergy(G)
#
#
#path=pathfromnodes(h)
#    
#fig = plt.figure()
#
#ax = fig.add_subplot(111, projection='3d')
#
#x=path['x'].values


#sort these dataframes in acsending order
xvalues=pd.DataFrame(unique_x).sort_values([0],ascending=True).reset_index(drop=True)
yvalues=pd.DataFrame(unique_y).sort_values([0],ascending=True).reset_index(drop=True)
zvalues=pd.DataFrame(unique_z).sort_values([0],ascending=True).reset_index(drop=True)

    
    

#Ecdf=sorted_data[['x','y','z','Ec']].copy()
#Ecdf=Ecdf.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
#Ecarr=Ecdf.values
#dictEc=dict(enumerate(Ecarr, 1))
#G=nx.Graph()
#G.add_nodes_from(dictEc.keys())
#for key, n in G.nodes.items():
#    n['pos']=dictEc[key][0:3]
#    n['pot']=dictEc[key][3]
#        
#    

#def edgeweight(source,target,xvalues,yvalues,zvalues,Ecdf):

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
        print(key)
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
        
    print(key)
    

    #
##loop through different z values along the device
#for z in zvalues:
#    #extract x-y plane for a z value
#    zslice=extract_slice(df,'z',z, drop=True)
#    l=zslice[zslice['Ec'] == min(zslice['Ec'])]
#    l=l[['x','y', 'Ec']].copy()
#    d1={'z':z,'coords':l.values}

#    
#    center=nodetocoord(source,xvalues,yvalues,zvalues)
#    neighbour=nodetocoord(target,xvalues,yvalues,zvalues)
#    
#    distance=np.linalg.norm(np.array(center[0:3])-np.array(neighbour[0:3]))
#    potentialdiff=(Ecdf['Ec'].iloc[source]+Ecdf['Ec'].iloc[target])/2
#    
#    return distance*potentialdiff
##
##    
#for key, n in list(G.nodes.items())[:-1]:
#    xneighs=NNX(key,xvalues,yvalues,zvalues)
#    yneighs=NNY(key,xvalues,yvalues,zvalues)
#    zneighs=NNZ(key,xvalues,yvalues,zvalues)
#    
#    if key==xneighs[0]:
#       g=0
#    else:
#        G.add_edge(key,xneighs[0],weight=float(edgeweight(key,xneighs[0],xvalues,yvalues,zvalues,Ecdf)))
#    
#    if key==xneighs[1]:
#       g=0
#    else:
#
#        G.add_edge(key,xneighs[1],weight=float(edgeweight(key,xneighs[1],xvalues,yvalues,zvalues,Ecdf)))
#
#
#    if key==yneighs[0]:
#       g=0
#    else:
#        print(key)
#        G.add_edge(key,yneighs[0],weight=float(edgeweight(key,yneighs[0],xvalues,yvalues,zvalues,Ecdf)))
#    

#    if key==yneighs[1]:
#        g=0
#    else:
#        G.add_edge(key,yneighs[1],weight=float(edgeweight(key,yneighs[1],xvalues,yvalues,zvalues,Ecdf)))
#        
#    if key==zneighs[0]:
#       g=0
#    else:
#        G.add_edge(key,zneighs[0],weight=float(edgeweight(key,zneighs[0],xvalues,yvalues,zvalues,Ecdf)))
#   
#    if key==zneighs[1]:
#      g=0
#    else:
#        G.add_edge(key,zneighs[1],weight=float(edgeweight(key,zneighs[1],xvalues,yvalues,zvalues,Ecdf)))
#        
#    print(key)
#    #
###loop through different z values along the device
##for z in zvalues:
##    #extract x-y plane for a z value
##    zslice=extract_slice(df,'z',z, drop=True)
##    l=zslice[zslice['Ec'] == min(zslice['Ec'])]
##    l=l[['x','y', 'Ec']].copy()
##    d1={'z':z,'coords':l.values}
##    
##    for l in l.values:
##        d2=np.append(l,z)
##        coords=np.vstack((coords,d2))
##        
##    lvalues=lvalues.append(d1, ignore_index=True)
##
##    i=i+1
#

start=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == 0)]

end=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == zvalues.iloc[len(zvalues)-1][0])]
s=nx.shortest_path_length(G,start.index.values[0],end.index.values[0],weight='weight')
h=nx.shortest_path(G,start.index.values[0],end.index.values[0],weight='weight')

#
##coords[:, 2], coords[:, 3] = coords[:, 3], coords[:, 2].copy()
##sortedcoords=coords[coords[:,2].argsort()]
##sortedcoordsdf=pd.DataFrame(sortedcoords,columns=['x','y','z', 'Ec'])
##dictcoords=dict(enumerate(sortedcoords, 1))
##G=nx.Graph()
##G.add_nodes_from(dictcoords.keys())
##for key,n in G.nodes.items():
##    tup=dictcoords[key].tolist()
##    n['pos']=dictcoords[key][0:3]
##    n['pot']=dictcoords[key][3]
##
##idx=0
##for idx in range(len(zvalues)-1):
##    positions=sortedcoordsdf.index[sortedcoordsdf['z'] == zvalues[idx]].values
##    neighposition=sortedcoordsdf.index[sortedcoordsdf['z'] == zvalues[idx+1]].values
##    for nkey in neighposition:
##        for key in positions:
##            pos=G.node[key+1]['pos']
##            neighpos=G.node[nkey+1]['pos']
##            dist=np.linalg.norm(pos-neighpos)
##            G.add_edge(key+1,nkey+1, weight=dist)
##            G[key+1][nkey+1]['dist']=dist
##        
##    
#start=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == 0)]

#end=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == zvalues.iloc[len(zvalues)-1][0])]
#s=nx.shortest_path_length(G,start.index.values[0],end.index.values[0],weight='weight')
#h=nx.shortest_path(G,start.index.values[0],end.index.values[0],weight='weight')
##

for node in h:
    nodeweights=G.node[node]['pot']+nodeweights
#    
averagenodeenergy=nodeweights/len(h)

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


save_json('C:\\Users\\Clayton\\Desktop\\Guillaume\\graph.js',G)



k=read_json_file('C:\\Users\\Clayton\\Desktop\\Guillaume\\graph.json')

#directory="C:\\Users\\Clayton\\Desktop\\CNSI test"
#file='p_structure_0.17_10nm-out.vg_0.00.vd_-0.20.vs_0.00.unified'

##
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
#   
#    
#fig = plt.figure()
#
#ax = fig.add_subplot(111, projection='3d')
#
#x=path['x'].values
#
#y=path['y'].values
#
#z=path['z'].values
#
#ax.set_xlim(0, xvalues[0].iat[-1]) 
#ax.set_ylim(0,yvalues[0].iat[-1])
#ax.set_zlim(0,zvalues[0].iat[-1])
#
#ax.scatter(x, y, z, c='r', marker='o')
#
##directory="C:\\Users\\Clayton\\Desktop\\CNSI test"
##file='p_structure_0.17_10nm-out.vg_0.00.vd_-0.20.vs_0.00.unified'
###
##directory ='C:\\Users\\Clayton\\Desktop\\10nmAlGaN\\Bias10'
##file= 'p_structure_0.17_10nm-out.vg_0.00.vd_0.00.vs_0.00.unified'

#directory ='C:\\Users\\Clayton\\Desktop\\30nmAlGaN\\Bias8'
#file= 'p_structure_0.17_30nm-out.vg_0.00.vd_-0.20.vs_0.00.unified'

#os.chdir(directory)
##df=pd.read_csv(file, delimiter=',')
#df=pd.read_csv(file, delimiter=',')
#g=lowestpoint(df)
##

#
#y=path['y'].values
#
#z=path['z'].values
#
#ax.set_xlim(0, xvalues[0].iat[-1]) 
#ax.set_ylim(0,yvalues[0].iat[-1])
#ax.set_zlim(0,zvalues[0].iat[-1])
#
#ax.scatter(x, y, z, c='r', marker='o')
#
#
#save_json('C:\\Users\\Clayton\\Desktop\\Guillaume\\graph.js',G)
#
#
#
#k=read_json_file('C:\\Users\\Clayton\\Desktop\\Guillaume\\graph.json')



