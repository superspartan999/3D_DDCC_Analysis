# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 00:01:01 2022

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

# KERNEL = np.array([[1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1],
#                    [1, 1, 0, 1, 1],
#                    [1, 1, 1, 1, 1],
#                    [1,1, 1, 1, 1, 1,1],
#                    [1,1, 1, 1, 1, 1,1]], dtype=np.int8)

KERNEL = np.array([[1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 0, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]], dtype=np.int8)


KERNEL = np.array([[ 1, 1, 1],
                    [ 1, 0, 1],
                    [ 1, 1, 1]], dtype=np.int8)

l=3
KERNEL=np.ones(shape=(l,l))
KERNEL[int(np.ceil(l/2-1)),int(np.ceil(l/2-1))]=0

def evolve(M,init_b,init_r):

    kws = dict(mode='same', boundary='wrap')
    b_neighs = convolve2d(M == init_b, KERNEL, **kws)
    r_neighs = convolve2d(M == init_r, KERNEL, **kws)
    neighs   = convolve2d(M !=-1,  KERNEL, **kws)
    
    b_dissatisfied = (b_neighs / neighs < SIM_T) & (M == init_b)
    b_satisfied=(b_neighs / neighs > SIM_T) & (M == init_b)
    r_dissatisfied = (r_neighs / neighs < SIM_T) & (M == init_r)
    r_satisfied=(r_neighs / neighs > SIM_T) & (M == init_r)
    

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
   
iterations=100
con_table=pd.DataFrame()

cg_full_ratio_list=np.array([])
fullratiolist=pd.DataFrame(columns=['nb','nr','nz'])
for i in range(iterations):

    N = 50    # Grid will be N x N
    SIM_T = 0.3  # Similarity threshold (that is 1-τ)
    
    
    B_to_R = 0.5   # Ratio of blue to red people
    init_b=200
    init_r=100
    M=rand_init(N, B_to_R,init_b,init_r)
    
    # # checkerboard alloy
    # M=np.indices((N,N)).sum(axis=0)%2
    # M=np.where(M==1,init_b,M)
    # M=np.where(M==0,init_r,M)
    
    # #columnar alloy
    # a=[True,False]
    # M=np.tile(a,(N,N))
    # M=M[:,:51]
    # M=np.where(M==1,init_b,M)
    # M=np.where(M==0,init_r,M)
    
    
    count1=np.count_nonzero(M==init_b)
    # plt.figure(1)
    
    # # plt.imshow(M)
    # plt.colorbar()
    # plt.clim(0,200)
    timestep=5
    for i in range(0,timestep):
        evolve(M,init_b,init_r)
    
    
    # # plt.colorbar()
    # create an array full of "False"
    bool_arr = np.full((N,N), False)
    
    # create a list of randomly picked indices, one for each row
    idx = np.random.randint(N, size=N)
    
    # replace "False" by "True" at given indices
    bool_arr[range(N), idx] = True
    # Array for random sampling
    # sample_arr = [True, False]
    p=1
    bool_arr=np.random.choice(a=[True, False], size=(N, N), p=[p, 1-p])
    # # Create a 2D numpy array or matrix of 3 rows & 4 columns with random True or False values
    sample_arr=[True,False]
    bool_arr = np.random.choice(sample_arr, size=(N,N))

    
    
    random_sample=M.copy()
    sample_coords=np.array(np.where(bool_arr==False)).T
    
    # for coord in sample_coords:
    #     random_sample[coord[0],coord[1]]=0
        
    atom_stream=random_sample.flatten()
    atom_stream=np.random.choice(atom_stream, replace=False,size=int(atom_stream.size * p))
    # atom_stream[indices]=0
    # atom_stream= np.random.choice(atom_stream, size=int(p*len(atom_stream)))
    block_num=25
    block_list=np.split(atom_stream,block_num)
    block_size=len(block_list[1])   
    ratiolist=pd.DataFrame({'nb':np.zeros(len(block_list)),'nr':np.zeros(len(block_list)),'nz':np.zeros(len(block_list))})
    
    for i,block in enumerate(block_list):
        nb=np.count_nonzero(block==init_b)
        nr=np.count_nonzero(block==init_r)
        nz=np.count_nonzero(block)
        nb_ratio=nb/block_size
        nr_ratio=nr/block_size
        n_non_zero_ratio=nz/block_size
        ratiolist['nb'].iloc[i]=nb
        ratiolist['nr'].iloc[i]=nr
        ratiolist['nz'].iloc[i]=nz
    
    fullratiolist=pd.concat([fullratiolist,ratiolist])

fullratiolist['nbratio']=fullratiolist['nb']/(fullratiolist['nb']+fullratiolist['nr'])
fullratiolist['nrratio']=fullratiolist['nr']/(fullratiolist['nb']+fullratiolist['nr'])
expected=len(fullratiolist)*(B_to_R)**(block_size)
class_size=100
expected_list=np.array([expected])
for i in np.linspace(1,class_size-1,class_size-1):
    expected=expected*(B_to_R)/(1-(B_to_R))*(block_size+1-i)/i
    expected_list=np.append(expected_list,expected)
    
con_table=np.histogram2d(fullratiolist['nb'],fullratiolist['nr'],bins=5)
# cg_random_sample=random_sample.copy()
# # plt.figure(1)     
    # plt.figure(3)
# cluster_generator=np.histogram(full_ratio_list,bins=20)