# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 21:17:15 2022

@author: me_hi
"""

import numpy as np
import random as rd

import os
import pandas as pd
from functions import *

import simplejson as json
from matplotlib import cm

    
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//int(nrows), int(nrows), -1, int(ncols))
               .swapaxes(1,2)
               .reshape(-1, int(nrows), int(ncols)))    
comp=0.5
# directory='D:\\Research\\Simulation Data\\n_type_AlGaN_0.30_40nm'

# directory = 'H:\\My Drive\\30nmAlGaN32\\Bias2'
# file='AlGaN_32_30nm_-out.vg_0.00.vd_0.00.vs_0.00.unified'

file='APT_In_map_0.5.csv'

# os.chdir(directory)
# df=pd.read_csv(file, delimiter=',')

os.chdir('C:\\Users\\me_hi\\Downloads')
df=pd.read_csv('APT_In_map_0.5.csv', delimiter=",", header=None)

node_map=df[['x','y','z']].copy()
#round up values in node map to prevent floating point errors
rounded_nodes=node_map.round(decimals=10)
#
##sort the nodes in ascending order
sorted_nodes=rounded_nodes.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
sorted_data=df.round({'x':10,'y':10,'z':10})
sorted_data=sorted_data.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)

#sorted_data=sorted_data[(sorted_data['z']>3e-6) & (sorted_data['z']<7e-6)]
#sorted_data=sorted_data[sorted_data['z']<5.6e-6]

#create dataframes for each xyz dimension in the mesh. this creates a dimension list 
#that gives us the total no. of grid points in any given direction
unique_x=sorted_data['x'].unique()
unique_y=sorted_data['y'].unique()
unique_z=sorted_data['z'].unique()


#sort these dataframes in acsending order
xvalues=pd.DataFrame(unique_x).sort_values([0],ascending=True).reset_index(drop=True)
yvalues=pd.DataFrame(unique_y).sort_values([0],ascending=True).reset_index(drop=True)
zvalues=pd.DataFrame(unique_z).sort_values([0],ascending=True).reset_index(drop=True)

# for 
comp_av=av_values(sorted_data,'Comp')
comp_av=comp_av.loc[comp_av['Comp']>comp-0.001]

zvalues=zvalues.loc[zvalues[0]>comp_av['z'].iloc[0]]
zvalues=zvalues.loc[zvalues[0]<comp_av['z'].iloc[-1]]
# cross_section=extract_slice(sorted_data,'z',zvalues.iloc[71][0],drop=True)

gb_full_ratio_list=np.array([])
for z_ind in range(0,len(zvalues)-1):
    # print(z_ind)
    cross_section=extract_slice(sorted_data,'z',zvalues.iloc[z_ind][0],drop=True)


    list_comp=np.sort(cross_section['Comp'].values)
    index=int(len(list_comp)*comp)
    
    crit=list_comp[index]
    
    atom_map=cross_section.copy()
    for index, row in atom_map.iterrows():
        if atom_map['Comp'].loc[index] > crit:
            atom_map['Comp'].loc[index]=100
            
        else:
            
            atom_map['Comp'].loc[index]=200

    var='Comp'
    # surf=atom_map
    
    surf=atom_map
    factor=5
    zmap=surf[[surf.columns[1],surf.columns[2], var ]].reset_index().round({'x':10,'y':10,'z':10})
    
    
    x=zmap[zmap.columns[1]].values
    
    y=zmap[zmap.columns[2]].values
    
    z=zmap[var].values
    
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    
    X,Y=np.meshgrid(x_vals,y_vals)
    
    Ec_array = np.empty(x_vals.shape + y_vals.shape)

    Ec_array.fill(np.nan)
    
    Ec_array[x_idx, y_idx] = zmap[var].values

# # plt.imshow(Ec_array)
    N=len(Ec_array) 
    bool_arr = np.full((N,N), False)
    
    # create a list of randomly picked indices, one for each row
    idx = np.random.randint(N, size=N)
    
    # replace "False" by "True" at given indices
    bool_arr[range(N), idx] = True
    # Array for random sampling
    # sample_arr = [True, False]
    p=2/3
    bool_arr=np.random.choice(a=[True, False], size=(N, N), p=[p, 1-p])
    # # Create a 2D numpy array or matrix of 3 rows & 4 columns with random True or False values
    sample_arr=[True,False]
    bool_arr = np.random.choice(sample_arr, size=(N,N))

    random_sample=Ec_array.copy()
    sample_coords=np.array(np.where(bool_arr==False)).T
    
    for coord in sample_coords:
        random_sample[coord[0],coord[1]]=0

    random_sample_block=blockshaped(random_sample, N/3, N/3)
    # # plt.imshow(random_sample)
    
    ratiolist=np.zeros(len(random_sample_block)-1)
    for i in range(len(random_sample_block)-1):
        nbr=np.count_nonzero(random_sample_block[i]==200)
        nbb=np.count_nonzero(random_sample_block[i]==100)
        n_total=np.count_nonzero(random_sample_block[i])
        
        ratio=nbr/n_total
        ratiolist[i]=ratio
    gb_full_ratio_list=np.append(gb_full_ratio_list,ratiolist)

gb_random_sample=random_sample.copy()
gb_histogram=np.histogram(gb_full_ratio_list,bins=20)
# cmap=cm.viridis

# # if len(y_vals)==len(x_vals):
# #         fig = plt.figure()
# # else:
# #     fig = plt.figure(figsize=(len(y_vals)/30, len(x_vals)/30)) 
# CS=plt.contourf(y_vals/1e-7,x_vals/1e-7,Ec_array,100,cmap=cm.viridis) 
# cbar=plt.colorbar(orientation='horizontal')
# # ticks=np.linspace(surf[var].min()*100,surf[var].max()*100,5)
# cbar.set_ticks(ticks)
# cbar.ax.tick_params(labelsize=18)