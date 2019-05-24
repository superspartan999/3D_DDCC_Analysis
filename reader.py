# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:27:40 2019

@author: Clayton
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import os
import networkx as nx


def extract_slice(data, slice_var, slice_val, drop=False):

    """
    This function grabs a 2D slice of a 3D data set. The function can set the
    variable and value as an argument.
    """

    if type(data) is not pd.DataFrame or type(slice_var) is not str:
        print('Input parameters of incorrect type.')
        return

    print("Slicing data...")
    my_filter = data[slice_var] == slice_val
    slice_data = data[my_filter]

    if drop:
        slice_data = slice_data.drop(slice_var, axis=1)

    return slice_data


def electric_field_z(df1, Ecom):
    #find all the values of z and put them in a list
    zvalues = df1['z'].unique()
    cols={}
    #create dataframe for conduction band and valence band
    Evalues=pd.DataFrame(columns=['z',Ecom])

    i=0
    #loop through different z values along the device
    for z in zvalues:
        #extract x-y plane for a z value
        zslice=extract_slice(df1,'z',z, drop=True)

        
        #average
        averagezsliceE=zslice[Ecom].mean()
        d1={'z':z,Ecom:averagezsliceE}
        Evalues.loc[i]=d1
        i=i+1


    return Evalues

#function to plot band diagram
def band_diagram_z(df1):
    #find all the values of z and put them in a list
    zvalues = df1['z'].unique()
    cols={}
    #create dataframe for conduction band and valence band
    Ecvalues=pd.DataFrame(columns=['z','Ec'])
    Evvalues=pd.DataFrame(columns=['z','Ev'])
    i=0
    #loop through different z values along the device
    for z in zvalues:
        #extract x-y plane for a z value
        zslice=extract_slice(df1,'z',z, drop=True)
        
        #average
        averagezsliceEc=zslice['Ec'].mean()
        averagezsliceEv=zslice['Ev'].mean()
        d1={'z':z,'Ec':averagezsliceEc}
        d2={'z':z,'Ev':averagezsliceEv}
        Ecvalues.loc[i]=d1
        Evvalues.loc[i]=d2
        i=i+1



    return Ecvalues,Evvalues

def ksp_yen(graph, node_start, node_end, max_k=2):
    distances, previous = dijkstra(graph, node_start)

    A = [{'cost': distances[node_end], 
          'path': path(previous, node_start, node_end)}]
    B = []

    if not A[0]['path']: return A

    for k in range(1, max_k):
        for i in range(0, len(A[-1]['path']) - 1):
            node_spur = A[-1]['path'][i]
            path_root = A[-1]['path'][:i+1]

            edges_removed = []
            for path_k in A:
                curr_path = path_k['path']
                if len(curr_path) > i and path_root == curr_path[:i+1]:
                    cost = graph.remove_edge(curr_path[i], curr_path[i+1])
                    if cost == -1:
                        continue
                    edges_removed.append([curr_path[i], curr_path[i+1], cost])

            path_spur = dijkstra(graph, node_spur, node_end)

            if path_spur['path']:
                path_total = path_root[:-1] + path_spur['path']
                dist_total = distances[node_spur] + path_spur['cost']
                potential_k = {'cost': dist_total, 'path': path_total}

                if not (potential_k in B):
                    B.append(potential_k)

            for edge in edges_removed:
                graph.add_edge(edge[0], edge[1], edge[2])

        if len(B):
            B = sorted(B, key=itemgetter('cost'))
            A.append(B[0])
            B.pop(0)
        else:
            break

    return A
#

directory = 'E:\\50nmAlGaN\\Bias -42'
file= 'p_structure_0.17_50nm-out.vg_0.00.vd_-4.20.vs_0.00.unified'

directory = 'E:\\10nmAlGaN\\Bias -42'
directory = "/Users/claytonqwah/Documents/Google Drive/Research/Transport Structure Project/3D data/10nmAlGaN/Bias -42"
file= 'p_structure_0.17_10nm-out.vg_0.00.vd_-4.20.vs_0.00.unified'

os.chdir(directory)
df=pd.read_csv(file, delimiter=',')

#find all the values of z and put them in a list
zvalues = np.sort(df['z'].unique())

zvalues=zvalues[:-1]
coords=np.empty(4)

#create dataframe for conduction band and valence band
lvalues=pd.DataFrame(columns=['z','coords'])


i=0
#loop through different z values along the device
for z in zvalues:
    #extract x-y plane for a z value
    zslice=extract_slice(df,'z',z, drop=True)
    l=zslice[zslice['Ec'] == min(zslice['Ec'])]
    l=l[['x','y', 'Ec']].copy()
    d1={'z':z,'coords':l.values}
    
    for l in l.values:
        d2=np.append(l,z)
        coords=np.vstack((coords,d2))
        
    lvalues=lvalues.append(d1, ignore_index=True)

    i=i+1
coords[:, 2], coords[:, 3] = coords[:, 3], coords[:, 2].copy()
sortedcoords=coords[coords[:,2].argsort()]
sortedcoordsdf=pd.DataFrame(sortedcoords,columns=['x','y','z', 'Ec'])
dictcoords=dict(enumerate(sortedcoords, 1))
G=nx.Graph()
G.add_nodes_from(dictcoords.keys())
for key,n in G.nodes.items():
    tup=dictcoords[key].tolist()
    n['pos']=dictcoords[key][0:3]
    n['pot']=dictcoords[key][3]

idx=0
for idx in range(len(zvalues)-1):
    positions=sortedcoordsdf.index[sortedcoordsdf['z'] == zvalues[idx]].values
    neighposition=sortedcoordsdf.index[sortedcoordsdf['z'] == zvalues[idx+1]].values
    for nkey in neighposition:
        for key in positions:
            pos=G.node[key+1]['pos']
            neighpos=G.node[nkey+1]['pos']
            dist=np.linalg.norm(pos-neighpos)
            G.add_edge(key+1,nkey+1, weight=dist)
            G[key+1][nkey+1]['dist']=dist
        
    
s=nx.shortest_path_length(G,1,len(sortedcoords)-1,weight='dist')
h=nx.shortest_path(G,1,len(sortedcoords)-1,weight='dist')
p=nx.shortest_simple_paths(G,1,7231)



nodeweights=0

for node in h:
    nodeweights=G.node[node]['pot']+nodeweights
    
averagenodeenergy=nodeweights/len(h )

#directory="C:\\Users\\Clayton\\Desktop\\CNSI test"
#file='p_structure_0.17_10nm-out.vg_0.00.vd_-0.20.vs_0.00.unified'
##
#directory ='C:\\Users\\Clayton\\Desktop\\10nmAlGaN\\Bias10'
#file= 'p_structure_0.17_10nm-out.vg_0.00.vd_0.00.vs_0.00.unified'

#directory ='C:\\Users\\Clayton\\Desktop\\30nmAlGaN\\Bias8'
#file= 'p_structure_0.17_30nm-out.vg_0.00.vd_-0.20.vs_0.00.unified'

#os.chdir(directory)
##df=pd.read_csv(file, delimiter=',')
#df=pd.read_csv(file, delimiter=',')
#g=lowestpoint(df)
##

#
#
#    return Ecvalues,Evvalues 
#
#df=pd.read_csv('E:\\Google Drive\\Research\\AlGaN Unipolar Studies\\10nmAlGaN\\p_structure_0.17_10nm-out.vg_0.00.vd_-0.20.vs_0.00.unified', delimiter=' ')
#df=df.drop(['Unnamed: 0'], axis=1)

#Ecomponent='E'
#


