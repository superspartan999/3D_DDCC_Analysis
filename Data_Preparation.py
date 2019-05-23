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


directory = 'E:\\Google Drive\\Research\\Transport Structure Project\\3D data\\Bias10'

directory = 'E:\\50nmAlGaN\\Bias -42'

#directory = 'C:\\Users\\Clayton\\Google Drive\Research\\Transport Structure Project\\3D data\\Bias10'

os.chdir(directory)


def write_space_df(file, head_len=5):

    """
    This function is inteded to take in a 3D .msh file and output a Pandas
    DataFrame object that contains the x, y, and z coordinates indexed to their
    Node numbers.
    """

    # Check input parameter types.
    if type(file) is not str or type(head_len) is not int:
        print('Input parameters of incorrect type.')
        return

    # Only run method if the file type is a .msh
    if file.endswith('.msh'):
        print("Extracting 3D space coordination...")

        # Dynamically determines the node space size.
        data_info = pd.read_csv(file, nrows=head_len, header=None)
        num_nodes = int(data_info.iloc[head_len-1, 0])

        raw_data = pd.read_csv(file, skiprows=head_len,
                               nrows=num_nodes, header=None,
                               delim_whitespace=True,
                               names=['x', 'y', 'z'], engine='python')
        return raw_data
    else:
        print("Error! File extension is not correct!")
        return


def extract_data(file, head_len=12):

    """
    This function is the most general extractor that pulls energy bands,
    ionized dopants, and temperature. The extractors read output files and
    convert them into Pandas DataFrames containing the data as values and
    using the Node numbers as indices.
    """

    # Check input parameter types.
    if type(file) is not str or type(head_len) is not int:
        print('Input parameters of incorrect type.')
        return

    # Only run with file extension is correct and set the column header
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
        return

    print("Extrating " + head + " data...")
    data_info = pd.read_csv(file, nrows=head_len, header=None)
    num_nodes = int(data_info.iloc[head_len-1, 0])

    my_data = pd.read_csv(file, skiprows=head_len, nrows=num_nodes, 
                          delim_whitespace=True, header=None, names=[head], 
                          engine='python')
    return my_data


def extract_carriers(file, head_len=12):

    """
    This function extracts the free carrier concentrations. Since the .np file
    contains both holes and electrons as separate halves, the function grabs
    sequentially then combines them with a unified Node index.
    """

    # Check input parameter types.
    if type(file) is not str or type(head_len) is not int:
        print('Input parameters of incorrect type.')
        return

    # Only run method if the file type is a .np
    if file.endswith('.np'):

        data_info = pd.read_csv(file, nrows=head_len, header=None)
        num_nodes = int(data_info.iloc[head_len-1, 0])

        print("Extracting electron concentration...")
        ndat = pd.read_csv(file, skiprows=head_len, nrows=num_nodes,
                           header=None, names=['n'], delim_whitespace=True, 
                           engine='python')
        print("Extracting hole concentration...")
        pdat = pd.read_csv(file, skiprows=2*head_len+num_nodes-1,
                           nrows=num_nodes, header=None, names=['p'],
                           delim_whitespace=True, engine='python')

        output = pd.concat([ndat, pdat], axis=1, join='outer')

        return output
    else:
        print("Error! File extension is not correct!")
        return


def extract_recombination(file, head_len=12):

    """
    This function extracts the recombination rates from their corresponding
    files. The files are parsed slightly different which causes errors in the
    extractor
    """

    # Check input parameter types.
    if type(file) is not str or type(head_len) is not int:
        print('Input parameters of incorrect type.')
        return

    # Only run with file extension is correct and set the column header
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

    rec_data = pd.read_csv(file, header=None, skiprows=head_len,
                           nrows=num_nodes, delim_whitespace=True, 
                           names=[head], engine='python')
    
    return rec_data


def create_unified_data_file(model_ID, node_map):
    
    """
    This function creates a unified data file that contains all parameters 
    associated with a particular model.
    """
    
    if type(model_ID) is not str or type(node_map) is not pd.DataFrame:
        print('Input parameters of incorrect type.')
        return
    
    # The Node map is used as the base on which the other data sets are added.
    output_data = node_map
    
    # The success variable is used to prevent the function from adding data
    # when the file analyzed is not parseable.
    success = True

    for file in glob.glob(model_ID + '.*'):
        print('Analyzing ' + file + '. Please wait...')
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
    
    # Reorder the headers to be easier to read
    output_data = output_data[['x', 'y', 'z', 'Ec', 'Ev', 'Ef', 'NDA', 'n','p', 'Radiative', 'Non-Radiative', 'Auger','Temperature']]
    
    output_data.to_csv(model_ID + '.unified', index_label='Node')
    return output_data
        
#


node_map = write_space_df('p_structure_0.17_50nm.msh')



mydf = create_unified_data_file('p_structure_0.17_50nm-out.vg_0.00.vd_-4.20.vs_0.00', node_map)

