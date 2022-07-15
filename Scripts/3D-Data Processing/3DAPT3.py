# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:48:32 2022

@author: Clayton
"""
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import random
import pandas as pd

l=5
kernel=np.ones(shape=(l,l,l))
kernel[int(np.ceil(l/2-1)),int(np.ceil(l/2-1))]=0

def rand_init(width, length, B_to_R,init_b,init_r,init_empty,empty):
    """ Random system initialisation.
    BLUE  =  0
    RED   =  1
    """
    vacant= int(length*width*width*empty)
    population = length * width * width -vacant #population size
    
    blues = int(population *B_to_R) #number of blues
    reds = population - blues #number of reds
    M = np.full(length * width * width, init_b) #
    M[:int(reds)] = init_r
    M[-vacant:]=init_empty
    np.random.shuffle(M)
    return M.reshape(length,width,width)

# def rand_init(N, B_to_R, EMPTY):
#     """ Random system initialisation.
#     BLUE  =  0
#     RED   =  1
#     EMPTY = -1
#     """
#     vacant = int(N * N * EMPTY)
#     population = N * N - vacant
#     blues = int(population * 1 / (1 + 1/B_to_R))
#     reds = population - blues
#     M = np.zeros(N*N, dtype=np.int8)
#     M[:reds] = 1
#     M[-vacant:] = -1
#     np.random.shuffle(M)
#     return M.reshape(N,N)

N = 50    # Grid will be N x N
L=100 #height/length of the film in the z-direction

SIM_T = 0.4   # Similarity threshold (that is 1-τ)
empty=0.1

B_to_R = 0.5   # Ratio of blue to red people
init_b=0
init_r=1
init_empty=-1


M=rand_init(N, L, B_to_R, init_b, init_r,init_empty,empty)
iterations=100

for i in range(iterations):
    kws = dict(mode='same')
    iterations=30
    
    kws = dict(mode='same')
    b_neighs = convolve(M == init_b, kernel, **kws)
    r_neighs = convolve(M == init_r, kernel, **kws)
    neighs   = convolve(M != init_empty,  kernel, **kws)
    
    b_dissatified = (b_neighs / neighs < SIM_T) & (M == init_b)
    r_dissatified = (r_neighs / neighs < SIM_T) & (M == init_r)
    M[r_dissatified | b_dissatified] = init_empty
    vacant = (M == init_empty).sum()
    
    n_b_dissatified, n_r_dissatified = b_dissatified.sum(), r_dissatified.sum()
    filling = -np.ones(vacant, dtype=np.int8)
    filling[:n_b_dissatified] = init_b
    filling[n_b_dissatified:n_b_dissatified + n_r_dissatified] = init_r
    np.random.shuffle(filling)
    M[M==init_empty] = filling

# for i in range(iterations):

#     b_neighs = convolve(M == init_b, kernel, **kws)
#     r_neighs = convolve(M == init_r, kernel, **kws)
#     neighs   = convolve(M !=-1,  kernel, **kws)
#     b_dissatisfied = (b_neighs / neighs < SIM_T) & (M == init_b)
#     b_satisfied=(b_neighs / neighs > SIM_T) & (M == init_b)
#     r_dissatisfied = (r_neighs / neighs < SIM_T) & (M == init_r)
#     r_satisfied=(r_neighs / neighs > SIM_T) & (M == init_r)
    
    
    # bdcoords=np.array(np.where(b_dissatisfied)).T
    # bscoords=np.array(np.where(b_satisfied)).T
    # rdcoords=np.array(np.where(r_dissatisfied)).T
    # rscoords=np.array(np.where(r_satisfied)).T
    
    # rsdf=pd.DataFrame(rscoords, columns=['x','y','z'])
    # bsdf=pd.DataFrame(bscoords, columns=['x','y','z'])
    
    # for coord in bdcoords:
    #     #find a random red node and switch places with a disatisfied blue node from list
    #     idx=np.random.randint(len(rscoords)-1)
    #     random_rs=rscoords[idx]
    #     M[random_rs[0],random_rs[1],random_rs[2]], M[coord[0],coord[1],coord[2]] \
    #     = M[coord[0],coord[1],coord[2]],M[random_rs[0],random_rs[1],random_rs[2]]
            
            
    # #     M[coord[0],coord[1],coord[2]]=a
    # #     M[random_rs.values[0][0],random_rs.values[0][1],random_rs.values[0][2]]=b
        
    #     rscoords=np.delete(rscoords,idx,0)
        
    # for coord in rdcoords:
        
    #     idx=np.random.randint(len(rscoords)-1)
    #     random_bs=rscoords[idx]
    #     M[random_bs[0],random_bs[1],random_bs[2]], M[coord[0],coord[1],coord[2]] \
    #     = M[coord[0],coord[1],coord[2]],M[random_bs[0],random_bs[1],random_bs[2]]
        
    #     bscoords=np.delete(bscoords,idx,0)