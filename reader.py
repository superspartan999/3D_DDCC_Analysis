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



#
directory = 'E:\\50nmAlGaN\\Bias -42'
#
#directory= 'C:\\Users\\Clayton\\Google Drive\\Research\\Transport Structure Project\\3D data\\50nmAlGaN\\Bias -42'

#directory = "/Users/claytonqwah/Documents/Google Drive/Research/Transport Structure Project/3D data/10nmAlGaN/Bias -42"

file= 'p_structure_0.17_50nm-out.vg_0.00.vd_-4.20.vs_0.00.unified'



def checkFrameRows(raw_data):
    (num_rows, num_cols) = raw_data.shape
    node_max = raw_data.Node.max()
    if node_max != num_rows:
        print('Error! Node max value does not match number of rows!\n Node Max: ' + str(node_max) + '\n Row Max: ' + str(num_rows))
    else:
        return num_rows

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

def extractFieldData(directory, file):
    os.chdir(directory)
    raw_data = pd.read_csv(file)
    
    num_rows = checkFrameRows(raw_data)
    potential_data=pd.DataFrame(index=np.arange(num_rows), 
                                 columns=['x', 'y', 'z', 'Ec'])
    electric_data = pd.DataFrame(index=np.arange(num_rows), 
                                 columns=['Ex', 'Ey', 'Ez', '|E|'])

    mytable=pd.pivot_table(raw_data, 'Ec', index=['x', 'y'], columns='z')
    my_x = mytable.index.levels[0].values

    my_y = mytable.index.levels[1].values
    my_z = mytable.columns.values
    vals = mytable.values
#    grad = np.gradient(mytable.values, [my_x, my_y, my_z])
    
    return mytable

#TODO: get the nearest neighbors, check that node is not on boundary
def getNearestNeighbor(raw_data, node_num, x_thresh, y_thresh, z_thresh):
    node_x = raw_data.at[node_num-1, 'x']
    node_y = raw_data.at[node_num-1, 'y']
    node_z = raw_data.at[node_num-1, 'z']
    
    my_data = raw_data[['Node','x','y','z']]
    my_filter = (abs(raw_data.x - node_x) < x_thresh) & \
                (abs(raw_data.y - node_y) < y_thresh) & \
                (abs(raw_data.z - node_z) < z_thresh)
    neighborhood = my_data[my_filter]
    
    neighborhood['distance'] = neighborhood.apply(lambda row, node_x=node_x, node_y=node_y, node_z=node_z: \
                ((row.x-node_x)**2+(row.y-node_y)**2+(row.z-node_z)**2)**0.5, axis=1)
    neighborhood['delX'] = neighborhood.apply(lambda row, node_x=node_x: abs(row.x - node_x), axis=1)
    neighborhood['delY'] = neighborhood.apply(lambda row, node_y=node_y: abs(row.y - node_y), axis=1)
    neighborhood['delZ'] = neighborhood.apply(lambda row, node_z=node_z: abs(row.z - node_z), axis=1)
    
    neighborhood = (neighborhood.sort_values(by=['distance', 'delX']))[neighborhood.delX != 0]
    

    
    return neighborhood.set_index('Node')




#extract x,y,z coordinates and x,y,z indices. the indices indicate the position of the point along a specific dimension list
def nodetocoord(index,xvalues,yvalues,zvalues):
    
    #obtain x index, which is the position of the value in the x-dimension list
    x_idx=int(floor(index/(len(yvalues)*len(zvalues))))
    
    x=xvalues.loc[x_idx][0]
    
    y_idx=int(floor(((index/len(zvalues))%len(yvalues))))
    
    y=yvalues.loc[y_idx][0]
    
    z_idx=int(floor(index%len(zvalues)))
    
    z=zvalues.loc[z_idx][0]
    
    return float(x) , float(y) , float(z) , x_idx, y_idx, z_idx

def coordtonode(x_idx,y_idx,z_idx,unique_x,unique_y,unique_z):
    
    max_x=len(unique_x)
    max_y=len(unique_y)
    max_z=len(unique_z)
    
    index = x_idx * max_y * max_z + y_idx * max_z + z_idx
    return index
    
    
def NNX(index,x_values,y_values,z_values):
    
    m=nodetocoord(index,x_values,y_values,z_values)
    
    x_idx=m[3]

    x_neg=x_idx-1
    
    x_pos=x_idx+1
    
    if x_neg < 0:
        
        x_neg=x_idx
    
    if x_pos >len(x_values)-1:
        
        x_pos=x_idx
    
    x_neg_node=coordtonode(x_neg,m[4],m[5],x_values,y_values,z_values)
    
    x_pos_node=coordtonode(x_pos,m[4],m[5],x_values,y_values,z_values)
    
    
    
    return x_neg_node,x_pos_node

def NNY(index,x_values,y_values,z_values):
    
    m=nodetocoord(index,x_values,y_values,z_values)
    
    y_idx=m[4]

    y_neg=y_idx-1
    
    y_pos=y_idx+1
    
    if y_neg < 0:
        
        y_neg=y_idx
    
    if y_pos >len(y_values)-1:
        
        y_pos=y_idx
    
    y_neg_node=coordtonode(m[3],y_neg,m[5],x_values,y_values,z_values)
    
    y_pos_node=coordtonode(m[3],y_pos,m[5],x_values,y_values,z_values)
    
    
    
    return y_neg_node,y_pos_node


def NNZ(index,x_values,y_values,z_values):
    
    m=nodetocoord(index,x_values,y_values,z_values)
    
    z_idx=m[5]
    
    

    z_neg=z_idx-1
    
    z_pos=z_idx+1
    
    if z_neg < 0: 
        
        z_neg=z_idx
    
    if z_pos >len(z_values)-1:
        
        z_pos=z_idx
         
    z_neg_node=coordtonode(m[3],m[4],z_neg,x_values,y_values,z_values)
    
    z_pos_node=coordtonode(m[3],m[4],z_pos,x_values,y_values,z_values)
    
    return z_neg_node,z_pos_node

def E_field(index,xvalues,yvalues,zvalues,sorted_data):

    
    X_NN=NNX(index,xvalues,yvalues,zvalues)
    Y_NN=NNY(index,xvalues,yvalues,zvalues)
    Z_NN=NNZ(index,xvalues,yvalues,zvalues)




    E_X=sorted_data.iloc[X_NN[1]]['Ec']-sorted_data.iloc[X_NN[0]]['Ec']/(sorted_data.iloc[X_NN[1]]['x']-sorted_data.iloc[X_NN[0]]['x'])
    E_Y=sorted_data.iloc[Y_NN[1]]['Ec']-sorted_data.iloc[Y_NN[0]]['Ec']/(sorted_data.iloc[Y_NN[1]]['y']-sorted_data.iloc[Y_NN[0]]['y'])
    E_Z=sorted_data.iloc[Z_NN[1]]['Ec']-sorted_data.iloc[Z_NN[0]]['Ec']/(sorted_data.iloc[Z_NN[1]]['z']-sorted_data.iloc[Z_NN[0]]['z'])
    
    E=np.sqrt(E_X*E_X+E_Y*E_Y+E_Z*E_Z)
    
    return E ,E_X, E_Y, E_Z

  

def Neighbourhood(index,xvalues,yvalues,zvalues):
    xneighs=NNX(index,xvalues,yvalues,zvalues)
    yneighs=NNY(index,xvalues,yvalues,zvalues)
    zneighs=NNZ(index,xvalues,yvalues,zvalues)
    
    center=nodetocoord(index,xvalues,yvalues,zvalues)
    xmin=nodetocoord(xneighs[0],xvalues,yvalues,zvalues)
    xplus=nodetocoord(xneighs[1],xvalues,yvalues,zvalues)
    ymin=nodetocoord(yneighs[0],xvalues,yvalues,zvalues)
    yplus=nodetocoord(yneighs[1],xvalues,yvalues,zvalues)
    zmin=nodetocoord(zneighs[0],xvalues,yvalues,zvalues)
    zplus=nodetocoord(zneighs[1],xvalues,yvalues,zvalues)
    
    
    nn=pd.DataFrame([center,xmin,xplus,ymin,yplus,zmin,zplus],columns=('x','y','z','xn','yn', 'zn'))
#    nn={'c':center[0:3],'x+':xplus[0:3],'x-':xmin[0:3],'y+':yplus[0:3],'y-':ymin[0:3],'z+':zplus[0:3],'z-':zmin[0:3]}
    
     
    return nn



directory = 'E:\\50nmAlGaN\\Bias -42'
file= 'p_structure_0.17_50nm-out.vg_0.00.vd_-4.20.vs_0.00.unified'


directory= 'C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume'

#directory= 'C:\\Users\\Clayton\\Desktop\\50nmAlGaN\\Bias -42'

#directory = "/Users/claytonqwah/Documents/Google Drive/Research/Transport Structure Project/3D data/10nmAlGaN/Bias -42"
file= 'LED4In-out.vg_0.00.vd_3.20.vs_0.00.unified'


os.chdir(directory)
df=pd.read_csv(file, delimiter=',')

#find all the values of z and put them in a list
zvalues = np.sort(df['z'].unique())

zvalues=zvalues[:-1]
coords=np.empty(4)

#create dataframe for conduction band and valence band
lvalues=pd.DataFrame(columns=['z','coords'])

#extract x,y,z coordinates and x,y,z indices. the indices indicate the position of the point along a specific dimension list


node_map=df[['x','y','z']].copy()
#round up values in node map to prevent floating point errors
rounded_nodes=node_map.round(decimals=10)
#
##sort the nodes in ascending order
sorted_nodes=rounded_nodes.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
sorted_data=df.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
sorted_data=sorted_data.round({'x':10,'y':10,'z':10})

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
    n['pos']=dictEc[key][0:3]
    n['pot']=dictEc[key][3]
        
    
def edgeweight(source,target,xvalues,yvalues,zvalues,Ecdf):
    
    center=nodetocoord(source,xvalues,yvalues,zvalues)
    neighbour=nodetocoord(target,xvalues,yvalues,zvalues)
    
    distance=np.linalg.norm(np.array(center[0:3])-np.array(neighbour[0:3]))
    potentialdiff=(Ecdf['Ec'].iloc[source]+Ecdf['Ec'].iloc[target])/2
    
    return distance*potentialdiff
#
#    
for key, n in list(G.nodes.items()):
    xneighs=NNX(key,xvalues,yvalues,zvalues)
    yneighs=NNY(key,xvalues,yvalues,zvalues)
    zneighs=NNZ(key,xvalues,yvalues,zvalues)
    

    if key==xneighs[0]:
        continue
    else:
        G.add_edge(key,xneighs[0],weight=float(edgeweight(key,xneighs[0],xvalues,yvalues,zvalues,Ecdf)))
    if key==xneighs[1]:
        continue
    else:
        G.add_edge(key,xneighs[1],weight=float(edgeweight(key,xneighs[1],xvalues,yvalues,zvalues,Ecdf)))

    if key==yneighs[0]:
        continue
    else:
        G.add_edge(key,yneighs[0],weight=float(edgeweight(key,yneighs[0],xvalues,yvalues,zvalues,Ecdf)))
    if key==yneighs[1]:
        continue
    else:
        G.add_edge(key,yneighs[1],weight=float(edgeweight(key,yneighs[1],xvalues,yvalues,zvalues,Ecdf)))
        
    if key==zneighs[0]:
        continue
    else:
        G.add_edge(key,zneighs[0],weight=float(edgeweight(key,zneighs[0],xvalues,yvalues,zvalues,Ecdf)))
    if key==yneighs[1]:
        continue
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
#    for l in l.values:
#        d2=np.append(l,z)
#        coords=np.vstack((coords,d2))
#        
#    lvalues=lvalues.append(d1, ignore_index=True)
#
#    i=i+1


#coords[:, 2], coords[:, 3] = coords[:, 3], coords[:, 2].copy()
#sortedcoords=coords[coords[:,2].argsort()]
#sortedcoordsdf=pd.DataFrame(sortedcoords,columns=['x','y','z', 'Ec'])
#dictcoords=dict(enumerate(sortedcoords, 1))
#G=nx.Graph()
#G.add_nodes_from(dictcoords.keys())
#for key,n in G.nodes.items():
#    tup=dictcoords[key].tolist()
#    n['pos']=dictcoords[key][0:3]
#    n['pot']=dictcoords[key][3]
#
#idx=0
#for idx in range(len(zvalues)-1):
#    positions=sortedcoordsdf.index[sortedcoordsdf['z'] == zvalues[idx]].values
#    neighposition=sortedcoordsdf.index[sortedcoordsdf['z'] == zvalues[idx+1]].values
#    for nkey in neighposition:
#        for key in positions:
#            pos=G.node[key+1]['pos']
#            neighpos=G.node[nkey+1]['pos']
#            dist=np.linalg.norm(pos-neighpos)
#            G.add_edge(key+1,nkey+1, weight=dist)
#            G[key+1][nkey+1]['dist']=dist
#        
#    
start=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[len(xvalues)/2][0])&(sorted_data['y'] == yvalues.iloc[len(yvalues)/2][0])&(sorted_data['z'] == 0)]

end=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[len(xvalues)/2][0])&(sorted_data['y'] == yvalues.iloc[len(yvalues)/2][0])&(sorted_data['z'] == zvalues.iloc[len(zvalues)-1][0])]
s=nx.shortest_path_length(G,start.index.values[0],end.index.values[0],weight='weight')
h=nx.shortest_path(G,start.index.values[0],end.index.values[0],weight='weight')

#
#
nodeweights=0
#
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


