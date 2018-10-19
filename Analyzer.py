# Analyzer is responsible for the modification of data files such that
# they can be read by our Python code.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 

__author__ = "Christian Robertson, Guillaume Lheureux, Clayton Qwah"
__copyright__ = "Copyright 2018"
__credits__ = ["Christian Robertson", "Guillaume Lheureux", "Clayton Qwah"]

__license__ = "GPL"
__version__ = "3.0.0"
__maintainer__ = "Christian Robertson"
__email__ = "09baylessc@gmail.com"
__status__ = "Development"


directory = 'E:\\HoletransportAlGaN_0.17_30nm\\Bias4'
file = 'p_structure_0.17_30nm-out.vg_0.00.vd_0.00.vs_0.00.unified'
#directory = 'E:\\HoletransportAlGaN_0.17_30nm\\Bias2'
#file = 'p_structure_0.17_30nm-out.vg_0.00.vd_-2.00.vs_0.00.unified'


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

os.chdir(directory)
my_data=pd.read_csv(file)    
num_rows = checkFrameRows(my_data)
EcEv=my_data[['x','y','z','Ec', 'Ev']]

#
#tree=scp.spatial.cKDTree(node_map)
#dd, ii=tree.query(node_map,7)
#
#j=0
#neighbourhood_dis=pd.DataFrame(columns=['dx', 'dy', 'dz'])
#three_d=plt.figure().gca(projection='3d')
#temp = getNearestNeighbor(my_data, 100, 1e-6, 1e-6, 1e-6)
#nxyz=pd.DataFrame(columns=['n','x', 'y', 'z'])
#p=ii[100]
#i=1
#for i in range(len(p)):
#    temp={'x' : EcEv.iloc[p[i]]['x'],'y' : EcEv.iloc[p[i]]['y'],'z' : EcEv.iloc[p[i]]['z']}
#    nxyz=nxyz.append(temp,ignore_index=True)
#
#three_d.scatter(nxyz['x'],nxyz['y'],nxyz['z'])
#plt.show()
#for n in ii:
#    for i in n
#        temp={'dx':(EcEv.iloc[n[0]]['x']-EcEv.iloc[n[1]]['x']),'dy':(EcEv.iloc[n[0]]['y']-EcEv.iloc[n[1]]['y']), 'dz':(EcEv.iloc[n[0]]['z']-EcEv.iloc[n[1]]['z'])}
#    
#    
#    neighbourhood=[EcEv.iloc[n[1]],EcEv.iloc[n[2]],EcEv.iloc[n[3]],EcEv.iloc[n[4]],EcEv.iloc[n[5]],EcEv.iloc[n[6]]])

    

#max values
max_x=mydf.loc[mydf['x'].idxmax()]['x']
max_y=mydf.loc[mydf['y'].idxmax()]['y']
max_z=mydf.loc[mydf['z'].idxmax()]['z']

#new_index=node_map['x']+node_map['y']*len(unique_x)+node_map['z']*len(unique_x)*len(unique_y)


my_data=pd.read_csv(file)



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



rounded_nodes=node_map.round(decimals=10)
sorted_nodes=rounded_nodes.sort_values(['x','y','z'],ascending=[True,True,True])

unique_x=rounded_nodes['x'].unique()
unique_y=rounded_nodes['y'].unique()
unique_z=rounded_nodes['z'].unique()



xvalues=pd.DataFrame(unique_x).sort_values([0],ascending=True).reset_index(drop=True)
yvalues=pd.DataFrame(unique_y).sort_values([0],ascending=True).reset_index(drop=True)
zvalues=pd.DataFrame(unique_z).sort_values([0],ascending=True).reset_index(drop=True)

def nodetocoord(index,rounded_nodes,unique_x,unique_y,unique_z):
    

    
    x_idx=index/(len(yvalues)*len(zvalues))
    
    x=rounded_x.iloc[x_idx]
    
    y_idx=(index/len(zvalues))%len(yvalues)
    
    y=rounded_y.iloc[y_idx]
    
    z_idx=index%len(rounded_z)
    
    z=rounded_z.iloc[z_idx]
    
    return float(x) , float(y) , float(z) , x_idx, y_idx, z_idx

def coordtonode(x,y,z,rounded_nodes,unique_x,unique_y,unique_z):
    
    max_x=len(unique_x)
    max_y=len(unique_y)
    max_z=len(unique_z)
    
    index = x * max_y * max_z + y * max_z + z
    return index
    
    
def NNX(index):
    
    x_neg=x_idx-1
    x_pos=x_idx+1
    
    return x_neg,x_pos

def NNY(y_idx):
    
    y_neg=y_idx-1
    y_pos=y_idx+1
    
    return y_neg,y_pos

def NNZ(z_idx):
    
    z_neg=z_idx-1
    z_pos=z_idx+1
    
    return z_neg,z_pos

def E(sorted_nodes,index):
    
    xdiff=rounded_x.iloc[x_pos]-rounded_x.iloc[x_neg]
    ydiff=rounded_x.iloc[y_pos]-rounded_x.iloc[y_neg]
    zdiff=rounded_x.iloc[y_pos]-rounded_x.iloc[y_neg]
    
    Ex=sorted_nodes.iloc(x_pos)['Ec']-sorted_nodes.iloc(x_neg)['Ec']
    Ey=sorted_nodes.iloc(y_pos)['Ec']-sorted_nodes.iloc(y_neg)['Ec']
    Ez=sorted_nodes.iloc(z_pos)['Ec']-sorted_nodes.iloc(z_neg)['Ec']
    
    
    
g=nodetocoord(14400,rounded_nodes,unique_x,unique_y,unique_z)
i=coordtonode(g[3],g[4],g[5],rounded_nodes,unique_x,unique_y,unique_z)

#axes = plt.gca()
#axes.set_xlabel('z(cm)')
#axes.set_ylabel('V(ev)')
#axes.set_xlim([0,12e-6])
#plt.scatter(banddiagram['z'], Ecvalues['Ec'])
#plt.scatter(Evvalues['z'], Evvalues['Ev']) 


#zvalues = rounded_nodes['z'].unique() 
#size_zslice=pd.DataFrame(columns=['z','size'])  
#j=0
#
#for z in zvalues:
#    zslice=extract_slice(rounded_nodes, 'z',0, drop=True)
#    size=len(zslice)
#    d={'z':z,'size':size}
#    size_zslice.loc[j]=d
#    j=j+1
        

##xlist=zslice['x'].tolist()
##ylist=zslice['y'].tolist()
##Eclist=zslice['Ec'].tolist()
##
##xx,yy=np.meshgrid(xlist,ylist)
##grad=(Eclist,xx,yy)
##mytab = extractFieldData(directory, file)
#
#my_data=pd.read_csv(file)
#temp = getNearestNeighbor(my_data, 100, 1e-7, 1e-7, 1e-7)
#num_rows = checkFrameRows(my_data)
#df1=my_data[['x','y','z','Ec']]
#v=df1.values

