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

from scipy.spatial import KDTree

directory = 'E:\\10nmAlGaN\\Bias -42'
directory = 'E:\\Google Drive\\Research\\Guillaume'
directory = 'C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume'
os.chdir(directory)


df=pd.read_csv("LED4In-out.vg_0.00.vd_3.20.vs_0.00.unified", delimiter=',')

sorted_data,xvalues,yvalues,zvalues=processdf(df)

zslice=extract_slice(sorted_data,'z',zvalues.iloc[len(zvalues)-1][0]/2,drop=True)

zmap=zslice[['x','y','Ec']].reset_index().round({'x':10,'y':10,'z':10})



def edgeweight2d(source,target,space,merged):
    
    average=(merged[source][2]+merged[target][2])
    print(average)

    
    return average



def coordtonode2d(x_idx,y_idx,unique_x,unique_y):
    
    max_y=len(unique_y)

    
    index = x_idx * max_y + y_idx 
    return index


def create_neighbors_array(index_matrix, x_array, y_array, z_array, is_4_axes_bool):
    around_x_mod, around_y_mod, around_size = select_around_mode (is_4_axes_bool)
    neighbors_array = np.empty((around_size+1,np.shape(z_array)[0]), dtype=int); neighbors_array.fill(-1)
    neighbor_pos = 1   
    for counter in range (np.shape(z_array)[0]):
        neighbor_pos = 1
        neighbors_array[0][counter] = 0
        for around_counter in range (around_size):
            new_x = x_array[counter] + around_x_mod[around_counter]
            if new_x >= 0 and new_x < np.shape(index_matrix)[0]:
                new_y = y_array[counter] + around_y_mod[around_counter]
                if new_y >= 0 and new_y < np.shape(index_matrix)[1]:
                    neighbors_array[int(neighbor_pos), int(counter)] = index_matrix[int(new_x),int(new_y)]
                    neighbor_pos += 1
                    neighbors_array[0, counter] += 1
    return neighbors_array

def select_around_mode (is_4_axes_bool):
    around_x_mod = ([0,1,0,-1,1,1,-1,-1])
    around_y_mod = ([1,0,-1,0,1,-1,-1,1])
    if is_4_axes_bool == True:
        around_size = 8
    else:
        around_size = 4
    return around_x_mod, around_y_mod, around_size

def generate_three_column_format(sorted_x_array, sorted_y_array, z_matrix):
    sorted_x_length = np.shape(sorted_x_array); sorted_y_length = np.shape(sorted_y_array)
    x_to_plot = np.empty((sorted_x_length[0]*sorted_y_length[0]))
    y_to_plot = np.empty((sorted_x_length[0]*sorted_y_length[0]))
    z_to_plot = np.empty((sorted_x_length[0]*sorted_y_length[0]))
    counter_to_plot = 0
    for counter_x in range(0, sorted_x_length[0]):
       for counter_y in range (0, sorted_y_length[0]):
           x_to_plot[counter_to_plot] = sorted_x_array[counter_x]
           y_to_plot[counter_to_plot] = sorted_y_array[counter_y]
           z_to_plot[counter_to_plot] = z_matrix[counter_x][counter_y]
           counter_to_plot+=1
    return x_to_plot, y_to_plot, z_to_plot

def get_index_array (input_array):
    for counter in range (numpy.shape(input_array)[0]):
        input_array[counter] = counter
    return input_array

def create_index_matrix(reference_matrix, x_array, y_array):
    index_matrix = np.empty(np.shape(reference_matrix))
    for counter in range (np.shape(x_array)[0]):
        index_matrix[int(x_array[counter])][int(y_array[counter])] = counter
    return index_matrix

def node_finder (input_matrix, x_array,y_array,z_array, is_4_axes_bool, x_values, y_values, flat_nodes_bool):
    print("Searching for nodes")
    input_matrix_shape = np.shape(input_matrix)
    x_array, y_array, z_array = generate_three_column_format(get_index_array(np.empty(input_matrix_shape[0])), get_index_array(np.empty(input_matrix_shape[1])), input_matrix)
    index_matrix = create_index_matrix(input_matrix, x_array, y_array)
    neighbors_array = create_neighbors_array(index_matrix, x_array, y_array, z_array, is_4_axes_bool)
    max_check = False
    min_check = False
    node_array = np.empty(np.shape(z_array)[0]); node_array.fill(-2)
    minima_counter = -1
    for global_counter in range (np.shape(z_array)[0]):
        if (max_check == True):
            max_check = False
        if (min_check == True):
            min_check = False
        for neighbor_counter in range (1,(neighbors_array[0][global_counter]+1).astype(int)):
            if (z_array[global_counter] > z_array[int(neighbors_array[neighbor_counter][global_counter])]):
                max_check = True
                break
            elif (z_array[global_counter] < z_array[neighbors_array[neighbor_counter][global_counter]] and flat_nodes_bool == False and min_check == False):
                min_check = True
                
        if (max_check == False and (flat_nodes_bool == True or min_check == True)):
        #if (max_check == 0):
            minima_counter += 1
            node_array[global_counter] = minima_counter

    minima_groups_counter = -1
    if (minima_counter<1):
        return 1, node_array, node_array, node_array, node_array ##This means there is 1 or 0 minima groups and, therefore, analysis cannot be made
    minima_groups_counter = minima_counter
    print(str(minima_groups_counter + 1) + " nodes found")
    if minima_groups_counter < 1:
        return 1, minima_groups_counter, minima_groups_counter, minima_groups_counter, minima_groups_counter
    max_points_in_a_group = 1
    points_in_minima_groups = np.empty((minima_groups_counter+1,max_points_in_a_group), dtype = int); points_in_minima_groups.fill(-1)
    points_to_plot = np.empty((minima_groups_counter+1,3))
    points_to_plot_counter = 0
    for global_counter in range (np.shape(z_array)[0]):
        if (node_array[global_counter] > -1):
            points_to_plot[points_to_plot_counter][0] = x_values[int(x_array[global_counter])]
            points_to_plot[points_to_plot_counter][1] = y_values[int(y_array[global_counter])]
            points_to_plot[points_to_plot_counter][2] = node_array[global_counter]
            points_to_plot_counter += 1
            points_in_minima_groups[int(node_array[global_counter])][0] = global_counter
    node_matrix = np.zeros((minima_groups_counter+1,minima_groups_counter+1))
    return 0, points_to_plot, node_array, points_in_minima_groups, node_matrix
        
def insert_point_in_working_array(working_array, point_index, z_array, first_working_point, last_pos_in_working_array, occupied_points_array):
    working_array[last_pos_in_working_array+1][0] = point_index
    first_working_point = int(first_working_point)
    point_index = int(point_index)
    if (first_working_point == -1):
        first_working_point = 0
        last_pos_in_working_array = 0
    else:
        if (z_array[int(working_array[first_working_point][0])] >=  z_array[point_index]):
            working_array[first_working_point][1] = last_pos_in_working_array+1
            working_array[last_pos_in_working_array+1][2] = first_working_point
            first_working_point = last_pos_in_working_array+1
        else:
            current_pos = first_working_point
            while (current_pos != -1):
                if (z_array[int(working_array[current_pos][0])] >=  z_array[point_index]):
                    working_array[last_pos_in_working_array+1][1] = working_array[current_pos][1]
                    working_array[last_pos_in_working_array+1][2] = current_pos
                    working_array[int(working_array[current_pos][1])][2] = last_pos_in_working_array+1   
                    working_array[current_pos][1] = last_pos_in_working_array+1
                    break
                if (working_array[current_pos][2] == -1):
                    working_array[current_pos][2] = last_pos_in_working_array+1
                    working_array[last_pos_in_working_array+1][1] = current_pos
                    working_array[last_pos_in_working_array+1][2] = -1
                    break
                current_pos = int(working_array[current_pos][2])
    last_pos_in_working_array += 1
    occupied_points_array[int(point_index)] = -2
    return int(first_working_point), int(last_pos_in_working_array)



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
for key, n in list(G.nodes.items())[:-1]:

    n['pos']=dictm[key][0:2].tolist()
    n['pot']=dictm[key][2]


xy=zmap[['index','x','y']]
point=xy[['x','y']].values
point_tree=KDTree(point)


for key, n in list(G.nodes.items())[:-1]:
    
    neighbourhood=point_tree.query_ball_point(point[key], 6.05e-8)
    
    
    neighbourhood.remove(key)
    for neigh in neighbourhood:
        G.add_edge(key,neigh,weight=edgeweight2d(key,neigh,space,merged))

def k_shortest_paths(G, source, target, k, weight=None):
     return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


   
#shortestpaths=[]
#for path in k_shortest_paths(G, 1, 2600, 3, weight='weight'):
#    shortestpaths.append(shortestpaths)


h=nx.bellman_ford_path(G,26,
                2575,weight='weight')     


path=pd.DataFrame(index=range(len(h)),columns={'x','y'})
for i,val in enumerate(h):
        path.loc[i]=zmap.iloc[val][['x','y']]
nodeweights=0
#
for node in h:
    nodeweights=G.node[node]['pot']+nodeweights
#    
averagenodeenergy=nodeweights/len(h)
plt.scatter(path['x'],path['y'], s=0.5)




xx,yy=np.meshgrid(x_vals,y_vals)
zz=np.zeros_like(xx)

for xind, x in enumerate(x_vals):
    for yind, y in enumerate(y_vals):
        zz[xind][yind]=zmap['Ec'].iloc[coordtonode2d(xind,yind, x_vals,y_vals)] 

fig = plt.figure()
CS=plt.contourf(x_vals,y_vals,Ec_array,30,cmap=cm.plasma) 

CS2=plt.contour(x_vals,y_vals,Ec_array, colors='black',linewidths=0.5)

plt.scatter(path['x'],path['y'], s=0.5)
#plt.clabel(CS2)

#
#h4=k_shortest_paths(G,1,2600,50,weight='weight')
#path_list= {}
#
#for index,h in enumerate(h4):
#    path=pd.DataFrame(index=range(len(h)),columns={'x','y'})
#    for i,val in enumerate(h):
#            path.loc[i]=zmap.iloc[val][['x','y']]
#    path_list[index]=path
##    
#for i, path in path_list.items():
#    plt.scatter(path['x'],path['y'], s=0.5)
#
#cbar = plt.colorbar(CS)

#fig = plt.figure()
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
    

    

    
    
    