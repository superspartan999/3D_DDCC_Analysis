# Data_Preparation is responsible for the modification of data files such that
# they can be read by our Python code.

import os
import pandas as pd
import numpy as np
import glob

__author__ = "Christian Robertson, Guillaume Lheureux, Clayton Qwah"
__copyright__ = "Copyright 2018"
__credits__ = ["Christian Robertson", "Guillaume Lheureux", "Clayton Qwah"]

__license__ = "GPL"
__version__ = "3.0.0"
__maintainer__ = "Christian Robertson"
__email__ = "09baylessc@gmail.com"
__status__ = "Development"

# Sets the current directory to the data folder
directory = 'C:\\Users\\Christian\\Box\\3DDCC_Simu\\Raw_Data\\Dislocation\\No Dislocation'
os.chdir(directory)

# This function should convert the file containing the set of 
# node-to-coordinate associations to a better format for further analysis.
# The output DataFrame is indexed to the node numbers. This means that calling
# raw_data.iloc(node_number) will return the a Series with the x, y, and z
# coordinates of that point.

def write_space_df(file, head_len=5):
    if file.endswith('.msh'):
        print("Extracting 3D space coordination...")
        
        # Dynamically determines the node space size.
        data_info = pd.read_csv(file, nrows=head_len, header=None)
        num_nodes = int(data_info.iloc[head_len-1, 0])
        
        raw_data = pd.read_csv(file, skiprows=head_len, 
                               nrows=num_nodes, header=None, 
                               delim_whitespace=True, 
                               names=['x', 'y', 'z'], engine='python')
        print('Data Read!')
        return raw_data
    else:
        print("Error! File extension is not correct!")
        return

# This function parses the .out file (Ec) into a DataFrame that is returned.
def extract_data(file, head_len=11):
    
    if file.endswith('.out'):
        head = 'Ec'
    elif file.endswith('.ef'):
        head = 'Ef'
    elif file.endswith('.Ev'):
        head = 'Ev'
    elif file.endswith('.nda'):
        head = 'NDA'
    elif file.endswith('.T'):
        head = 'Temperature'
    else:
        print("Error! File extension is not correct!")
        head = 'Unknown'
    
    print("Extrating " + head + " data...")
    data_info = pd.read_csv(file, nrows=head_len, header=None)
    num_nodes = int(data_info.iloc[head_len-1, 0])
    
    my_data = pd.read_csv(file, skiprows=head_len, nrows=num_nodes, 
                          header=None, sep='  ', names=[head], engine='python')
    return my_data

def extract_carriers(file, head_len=11):
    if file.endswith('.np'):
        data_info = pd.read_csv(file, nrows=head_len, header=None)
        num_nodes = int(data_info.iloc[head_len-1, 0])
        
        ndat = pd.read_csv(file, skiprows=head_len, nrows=num_nodes, header=None,
                           names=['n'], sep='  ', engine='python')
        pdat = pd.read_csv(file, skiprows=2*head_len+num_nodes-1, nrows=num_nodes,
                           header=None, names=['p'], sep='  ', engine='python')
        
        output = pd.concat([ndat, pdat], axis=1, join='outer')
        return output
    else:
        print("Error! File extension is not correct!")
        return
    
def extract_recombination(file, head_len=11):
    if file.endswith('.nonRad'):
        head = 'Non-Radiative'
    elif file.endswith('.Auger'):
        head = 'Auger'
    elif file.endswith('.Rad'):
        head = 'Radiative'
    else:
        print("Error! File extension is incorrect!")
        return
    
    print("Extracting " + head + " recombination data...")
    data_info = pd.read_csv(file, nrows=head_len, header=None)
    num_nodes = int(data_info.iloc[head_len-1, 0])
    my_data = pd.read_csv(file, header=None, skiprows=head_len, nrows=num_nodes, 
                          delim_whitespace=True, names=[head], 
                          engine='python')
    return my_data

# This function takes a 2D slice of data along a coordinate equal to a constant
# value. For our purposes, we are taking a slice of the 3D data such that y==0.
def extract_slice(data, slice_var, slice_val, drop=False):
    print("Slicing data...")
    my_filter = data[slice_var] == slice_val
    slice_data = data[my_filter]
    if drop:
        slice_data = slice_data.drop(slice_var, axis=1)
    return slice_data

def create_unified_data_file(model_ID, node_map):
    
    output_data = node_map
    success = True

    for file in glob.glob(model_ID + '.*'):
        print('Analyzing ' + file + '. Please wait...\n\n')
        if file.endswith('.np'):
            my_data = extract_carriers(file)
        elif file.endswith(('.Auger', '.nonRad', '.Rad')):
            my_data = extract_recombination(file)
        elif file.endswith(('.out', '.ef', '.Ev', '.nda', '.T')):
            my_data = extract_data(file)
        else:
            print(file + ' is not parseable at this time.')
            success = False
        
        if success:
            output_data = pd.concat([output_data, my_data], axis=1, join='outer')

        success = True
        
    return output_data
        

node_map = write_space_df('dislocation_line_2.msh')
mydf = create_unified_data_file('dislocation_line_2-out.vg_0.00.vd_0.00.vs_0.00', node_map)