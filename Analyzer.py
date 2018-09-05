# Analyzer is responsible for the modification of data files such that
# they can be read by our Python code.

import os
import pandas as pd
import numpy as np

__author__ = "Christian Robertson, Guillaume Lheureux, Clayton Qwah"
__copyright__ = "Copyright 2018"
__credits__ = ["Christian Robertson", "Guillaume Lheureux", "Clayton Qwah"]

__license__ = "GPL"
__version__ = "3.0.0"
__maintainer__ = "Christian Robertson"
__email__ = "09baylessc@gmail.com"
__status__ = "Development"

directory = 'D:\\HoletransportAlGaN_0.17_30nm\\HoletransportAlGaN_0.17_30nm'
file = 'p_structure_0.17_30nm-out.vg_0.00.vd_1.00.vs_0.00.unified'

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

def calculateField(raw_data, node, variable='x'):
    val = raw_data.at[node, 'y']
    print('Slice Value: ' + str(val))
    finalData = extract_slice(raw_data, 'y', raw_data.at[node, 'y'])
    return(finalData)

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
os.chdir(directory)
my_data=pd.read_csv(file)    
num_rows = checkFrameRows(my_data)
df1=my_data[['x','y','z','Ec']]
v=df1.values
#df1=df1.sort_values(by='x')
#xlist= df1['x'].tolist()
#ylist= df1['y'].tolist()
#zlist= df1['z'].tolist()
#Eclist=df1['Ec'].tolist()
#
#dx = np.diff(xlist)
#dy = np.diff(ylist)
#dz = np.diff(zlist)
#
#
E=np.gradient(v)

Ed=pd.DataFrame(E[0],columns=['x','y','z','El'])
Ed=Ed.sort_values(by='z')
Ed.plot(x='z', y=['El'])

#x=np.array(xlist)
#y=np.array(ylist)
#z=np.array(zlist)
#Ec=np.array(Eclist)
#
#dx = np.diff(xlist)
#dy = np.diff(ylist)
#dz = np.diff(zlist)
#
#xx,yy,zz, Ecc=np.meshgrid(x, y, z,Ec,indexing='ij',sparse=True)


#
#zslice=extract_slice(df1, 'z', 2.621729100152412e-07
# , drop=True)
#xlist=zslice['x'].tolist()
#ylist=zslice['y'].tolist()
#Eclist=zslice['Ec'].tolist()
#
#xx,yy=np.meshgrid(xlist,ylist)
#grad=(Eclist,xx,yy)
#mytab = extractFieldData(directory, file)
