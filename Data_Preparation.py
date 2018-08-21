# Data_Preparation is responsible for the modification of data files such that
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


# This function should convert the file containing the set of 
# node-to-coordinate associations to a better format for further analysis.
# The output DataFrame is indexed to the node numbers. This means that calling
# raw_data.iloc(node_number) will return the a Series with the x, y, and z
# coordinates of that point.

def write_space_df(directory, file, skiptop=5, skipbottom=2875044):
    os.chdir(directory)
    print('Directory Changed! Target Directory: ' + directory)
    # TODO: Dynamically adjust skiprows and skipfooter to account for different meshes.
    # SkipFooter number calculated from total file rows - nodetotal
    raw_data = pd.read_csv(file, skiprows=skiptop, skipfooter=skipbottom, header=None,
                           delim_whitespace=True, index_col=0,
                           names=['x', 'y', 'z'], engine='python')
    print('Data Read!')
    return raw_data

# This function takes a 2D slice of data along a coordinate equal to a constant
# value. For our purposes, we are taking a slice of the 3D data such that y==0.
    
def extract_slice(data, slice_var, slice_val, drop=False):
    my_filter = data[slice_var] == slice_val
    slice_data = data[my_filter]
    if drop:
        slice_data = slice_data.drop(slice_var, axis=1)
    return slice_data

directory = 'C:\\Users\\Christian\\Documents\\Github\\DDCC_3D_Project'
file = 'My_Mesh.msh'

#df = write_space_df(directory, file)
#my_df = extract_slice(df, 'y', 0, drop=True)