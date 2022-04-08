# -*- coding: utf-8 -*-
"""

Created on Wed Mar  9 17:38:02 2022

@author: me_hi
"""

import pandas as pd
import os
from functions import *


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

os.chdir('C:\\Users\\me_hi\\Downloads')
dat=pd.read_csv('APT_In_map_0.5_30nm_gaussian.csv', header=None)

dat=dat.rename(columns={0:'x',1:'y',2:'z',3:'Comp'})
df=dat.iloc[::2]

node_map=df[['x','y','z']].copy()
rounded_nodes=node_map.round(decimals=10)
#
##sort the nodes in ascending order
sorted_nodes=rounded_nodes.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
sorted_data=df.round({'x':10,'y':10,'z':10})
sorted_data=sorted_data.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
sorted_data=sorted_data.loc[sorted_data['y']<1.40e-6]
sorted_data=sorted_data.loc[sorted_data['x']<1.40e-6]

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
comp=0.5

gb_full_ratio_list=np.array([])
z_ratio_list=np.array([])
for z_ind in range(0,len(zvalues)-1):
    # print(z_ind)
    cross_section=extract_slice(sorted_data,'z',zvalues.iloc[1][0],drop=True)
    
    
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
    zmap=surf[[surf.columns[0],surf.columns[1], var ]].reset_index().round({'x':10,'y':10,'z':10})
    
    
    x=zmap[zmap.columns[1]].values
    
    y=zmap[zmap.columns[2]].values


    
    z=zmap[var].values
    
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    
    X,Y=np.meshgrid(x_vals,y_vals)
    
    Ec_array = np.empty(x_vals.shape + y_vals.shape)
    
    Ec_array.fill(np.nan)
    
    Ec_array[x_idx, y_idx] = zmap[var].values
    
    # plt.imshow(Ec_array)
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

    random_sample_block=blockshaped(random_sample, N/5, N/5)
    # # plt.imshow(random_sample)
    
    ratiolist=np.zeros(len(random_sample_block)-1)
    zratiolist=np.zeros(len(random_sample_block)-1)
    for i in range(len(random_sample_block)-1):
        nbr=np.count_nonzero(random_sample_block[i]==200)
        nbb=np.count_nonzero(random_sample_block[i]==100)
        nz=np.sum(random_sample_block[i]==0)
        n_total=np.count_nonzero(random_sample_block[i])
        
        ratio=nbr/n_total
        ratioz=nz/n_total
        ratiolist[i]=ratio
        zratiolist[i]=ratioz
    gb_full_ratio_list=np.append(gb_full_ratio_list,ratiolist)
    z_ratio_list=np.append(z_ratio_list,zratiolist)
    
    
gb_random_sample=random_sample.copy()
gb_histogram=np.histogram(gb_full_ratio_list,bins=20)

