# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 21:20:38 2022

@author: me_hi
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 23:48:27 2022

@author: me_hi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:26:08 2022

@author: me_hi
"""


import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import random
import pandas as pd

import numpy as np

def rand_init(N, B_to_R,init_b,init_r):
    """ Random system initialisation.
    BLUE  =  0
    RED   =  1
    """
    
    population = N * N #population size
    
    blues = int(population *B_to_R) #number of blues
    reds = population - blues #number of reds
    M = np.full(N*N, init_b) #
    M[:int(reds)] = init_r
    np.random.shuffle(M)
    return M.reshape(N,N)

KERNEL = np.array([[1, 1, 1, 1, 1],
                   [1, 1, 0, 1, 1],
                   [1, 1, 1, 1, 1]], dtype=np.int8)


KERNEL = np.array([[ 1, 1, 1],
                   [ 1, 0, 1],
                   [ 1, 1, 1]], dtype=np.int8)
def evolve(M):

    kws = dict(mode='same', boundary='wrap')
    b_neighs = convolve2d(M == 200, KERNEL, **kws)
    r_neighs = convolve2d(M == 100, KERNEL, **kws)
    neighs   = convolve2d(M !=-1,  KERNEL, **kws)
    
    b_dissatisfied = (b_neighs / neighs < SIM_T) & (M == 200)
    b_satisfied=(b_neighs / neighs > SIM_T) & (M == 200)
    r_dissatisfied = (r_neighs / neighs < SIM_T) & (M == 100)
    r_satisfied=(r_neighs / neighs > SIM_T) & (M == 100)
    

    bdcoords=np.array(np.where(b_dissatisfied)).T
    bscoords=np.array(np.where(b_satisfied)).T
    rdcoords=np.array(np.where(r_dissatisfied)).T
    rscoords=np.array(np.where(r_satisfied)).T
    
    rsdf=pd.DataFrame(rscoords, columns=['x','y'])
    bsdf=pd.DataFrame(bscoords, columns=['x','y'])
    
    for coord in bdcoords:
        
        random_rs=rsdf.sample()
        a=M[random_rs.values[0][0],random_rs.values[0][1]]
        b=M[coord[0],coord[1]]
        
        
        M[coord[0],coord[1]]=a
        M[random_rs.values[0][0],random_rs.values[0][1]]=b
        
        rsdf=rsdf.drop(random_rs.index)
    
    for coord in rdcoords:
        
        random_bs=bsdf.sample()
        a=M[random_bs.values[0][0],random_bs.values[0][1]]
        b=M[coord[0],coord[1]]
        
        
        M[coord[0],coord[1]]=a
        M[random_bs.values[0][0],random_bs.values[0][1]]=b
        
        bsdf=bsdf.drop(random_bs.index)
    
    
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
   
iterations=277

cg_full_ratio_list=np.array([])
for i in range(iterations):

    N = 51    # Grid will be N x N
    SIM_T = 0.3  # Similarity threshold (that is 1-τ)
    
    
    B_to_R = 0.3   # Ratio of blue to red people
    
    M=rand_init(N, B_to_R,200,100)
    # M=np.indices((N,N)).sum(axis=0)%2
    # M=np.where(M==1,200,M)
    # M=np.where(M==0,100,M)
    count1=np.count_nonzero(M==200)
    # plt.figure(1)
    
    # plt.imshow(M)
    # plt.colorbar()
    # plt.clim(0,200)
    # timestep=10
    # for i in range(0,timestep):
    #     evolve(M)
    
    
    # # plt.colorbar()
    # create an array full of "False"
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

    
    
    random_sample=M.copy()
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
    cg_full_ratio_list=np.append(cg_full_ratio_list,ratiolist)


plt.figure(1)     
    # plt.figure(3)
# cluster_generator=np.histogram(full_ratio_list,bins=20)