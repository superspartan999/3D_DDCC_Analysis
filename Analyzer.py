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

directory = 'C:\\Users\\Christian\\Box\\3DDCC_Simu\\' + \
            'Raw_Data\\Dislocation\\No Dislocation'
file = 'dislocation_line_2-out.vg_0.00.vd_0.00.vs_0.00.unified'

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
    electric_data = pd.DataFrame(index=np.arange(num_rows), 
                                 columns=['Ex', 'Ey', 'Ez', '|E|'])
    
    mytable=pd.pivot_table(raw_data, 'Ec', index=['x', 'y'], columns='z')
    my_x = mytable.index.levels[0].values
    my_y = mytable.index.levels[1].values
    my_z = mytable.columns.values
    vals = mytable.values
#    grad = np.gradient(mytable.values, [my_x, my_y, my_z])
    
    return mytable

mytab = extractFieldData(directory, file)
