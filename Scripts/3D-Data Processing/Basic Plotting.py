# Analyzer is responsible for the modification of data files such that
# they can be read by our Python code.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from math import floor, sqrt

__author__ = "Christian Robertson, Guillaume Lheureux, Clayton Qwah"
__copyright__ = "Copyright 2018"
__credits__ = ["Christian Robertson", "Guillaume Lheureux", "Clayton Qwah"]

__license__ = "GPL"
__version__ = "3.0.0"
__maintainer__ = "Christian Robertson"
__email__ = "09baylessc@gmail.com"
__status__ = "Development"

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

#function to obtain in-plane averaged values. takes a dataframe and a string to denote the value that is chosen 
def inplane(df1,variable):
    #find all the values of z and put them in a list
    zvalues = df1['z'].unique()
    #create dataframe for z values and in plane average values
    df=pd.DataFrame(columns=['z',variable])
    #loop through different z values along the device
    for i, z in enumerate(zvalues):
        #extract x-y plane for a z value
        zslice=extract_slice(df1,'z',z, drop=True)
        
        #average the values in the plane and insert them into dataframe
        averagezsliceEc=zslice[variable].mean()
        d1={'z':z,variable:averagezsliceEc}
        df.loc[i]=d1
        


    return df

def plotsurf(surf,Var):

    Var='Comp'
    zmap=surf[[surf.columns[1],surf.columns[2], Var ]].reset_index().round({'x':10,'y':10,'z':10})
    
    
    x=zmap[zmap.columns[1]].values
    
    y=zmap[zmap.columns[2]].values
    
#    z=zmap[Var].values
    
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    
    Ec_array = np.empty(x_vals.shape + y_vals.shape)
    
    Ec_array.fill(np.nan)
    
    Ec_array[x_idx, y_idx] = zmap[Var].values
    
    cmap=cm.viridis
    
    fig = plt.figure(figsize=(len(y_vals)/20, len(x_vals)/20))
    plt.contourf(y_vals/10e-7,x_vals/10e-7,Ec_array,100,cmap=cmap) 
    
    plt.colorbar()


directory = 'D:\\Bias10'
file= 'n_type_AlGaN_0.14_13nm-out.vg_0.00.vd_2.80.vs_0.00.unified'


os.chdir(directory)

#read unified file
my_data=pd.read_csv(file, delimiter=',')


#max values
max_x=my_data.loc[my_data['x'].idxmax()]['x']
max_y=my_data.loc[my_data['y'].idxmax()]['y']
max_z=my_data.loc[my_data['z'].idxmax()]['z']


#extract node map of the data
node_map=my_data[['x','y','z']].copy()


#round up values in node map to prevent floating point errors
rounded_nodes=node_map.round(decimals=10)

#creates a new dataframe that sorts the nodes in ascending order
sorted_nodes=rounded_nodes.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
sorted_data=my_data.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
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

bottom_surf=extract_slice(sorted_data,'z',zvalues.iloc[0][0],drop=True)
top_surf=extract_slice(sorted_data,'z',zvalues.iloc[-1][0],drop=True)
cross_section=extract_slice(sorted_data,'z',zvalues.iloc[int((len(zvalues)-1)/2)][0],drop=True)
p_section=extract_slice(sorted_data,'z',9e-6,drop=True)
s1_surf=extract_slice(sorted_data,'x',xvalues.iloc[0][0],drop=True)
s2_surf=extract_slice(sorted_data,'x',xvalues.iloc[-1][0],drop=True)
s3_surf=extract_slice(sorted_data,'y',yvalues.iloc[0][0],drop=True)
s4_surf=extract_slice(sorted_data,'y',yvalues.iloc[-1][0],drop=True)


#create a dataframe for Ec values
Ec=inplane(sorted_data,'Ec')

#create a dataframe for Ev values
Ev=inplane(sorted_data,'Ev')

#plt.plot(Ec['z'],Ec['Ec'])
##plt.plot(Ev['z'],Ev['Ev'])
##create csv file from unified data
#filemake=sorted_data.to_csv(file,sep=',')
###
#plotsurf(s1_surf,'Comp')



