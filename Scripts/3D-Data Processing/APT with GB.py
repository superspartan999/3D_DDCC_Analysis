# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
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
=======
Created on Tue Feb 15 13:50:19 2022

@author: Clayton
"""

import pandas as pd
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from scipy.spatial import KDTree
from scipy.spatial import cKDTree
N=100

p=0.8
>>>>>>> 3a5cd370a28d27c3d1ba0ae3701730af5c94a5cf

dat=dat.rename(columns={0:'x',1:'y',2:'z',3:'Comp'})
df=dat.iloc[::2]

node_map=df[['x','y','z']].copy()
rounded_nodes=node_map.round(decimals=10)
#
##sort the nodes in ascending order
sorted_nodes=rounded_nodes.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
sorted_data=df.round({'x':10,'y':10,'z':10})
sorted_data=sorted_data.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
sorted_data=sorted_data.loc[sorted_data['y']<1.42e-6]
sorted_data=sorted_data.loc[sorted_data['x']<1.42e-6]

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

<<<<<<< HEAD
gb_full_ratio_list=np.array([])
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
=======
random_x_list = np.random.randint(0,100,size=N)
random_y_list = np.random.randint(0,100,size=N)


#def find_neighbours(alloy,x,y):
#    
#    

        PtoA=2/((diameter[0])*1e-4/2)
>>>>>>> 3a5cd370a28d27c3d1ba0ae3701730af5c94a5cf
    
    z=zmap[var].values
    
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    
    X,Y=np.meshgrid(x_vals,y_vals)
    
    Ec_array = np.empty(x_vals.shape + y_vals.shape)
    
    Ec_array.fill(np.nan)
    
    Ec_array[x_idx, y_idx] = zmap[var].values
    
    plt.imshow(Ec_array)
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

<<<<<<< HEAD
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
=======
    fig=plt.figure(7)
    plt.scatter(voltages,np.array(contribution)*100, label=size)
    plt.title('Percentage Contribution of J$_{perimeter}$ for 200 micron device')
    plt.xlabel('Voltage (V)')
    plt.yticks(np.arange(0, 100, step=10))
    plt.ylim(0,100)
    plt.xlim(-4,2.5)
    plt.ylabel('Percentage Contribution of J$_{perimeter}$ (%)')
    plt.grid()
    name=directory
    name=name.replace('G:\My Drive\Research\Transport Structure 2020\\', '')
       
    colors = plt.cm.jet(np.linspace(0,1,np.size(voltages)+1))
    poslist=[x for x in fits.keys() if x > 0]
    neglist=[x for x in fits.keys() if x < 0]
    neglist.sort(reverse=True)
    pos={}
    neg={}
    for i,key in enumerate(fits.keys()):   
        if key<0:
            neg[key]=fits[key]
    for i,key in enumerate(fits.keys()):  
        if key>0:
            pos[key]=fits[key]
    fig=plt.figure(3)  
    # neg=sorted(neg.items(),key=operator.itemgetter(1),reverse=True)
    for i,key in enumerate(poslist):   
            plt.plot(pos[key]['p/a'],abs(pos[key]['j']),label=str(key)+' V',color=colors[i])
            
    # for i,key in enumerate(neglist):   
    #         plt.semilogy(neg[key]['p/a'],abs(neg[key]['j']),label=str(key)+' V',linestyle='dashed',color=colors[i])
            
    plt.title('J vs P/A')
    plt.xlabel('P/A (cm$^{-1}$)')
    plt.ylabel('J (A/cm$^2$')
    fig=plt.figure(4)
    plt.plot(mc['V'],mc['Slope'],label=directory)
    plt.title('J Perimeter vs Volt')  
    plt.grid()
    plt.xlabel('Applied Bias (V)')
    plt.ylabel('$J_{perimeter}$ (A/cm)')    
    
    fig=plt.figure(5)
    
    plt.plot(mc['V'],mc['Intercept'],label=directory)   
    plt.title('J Diode vs Volt')  
    plt.xlabel('Applied Bias (V)')
    plt.grid()
    plt.ylabel('$J_{diode}$ (A/cm$^{2}$)') 
    
    filedict[name]=mc
#    

    
>>>>>>> 3a5cd370a28d27c3d1ba0ae3701730af5c94a5cf
