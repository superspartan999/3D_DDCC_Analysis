# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:26:08 2022

@author: me_hi
"""


import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import random


import numpy as np

def rand_init(N, B_to_R, EMPTY):
    """ Random system initialisation.
    BLUE  =  0
    RED   =  1
    EMPTY = -1
    """
    vacant = N * N * EMPTY
    population = N * N - vacant
    blues = int(population *B_to_R)
    reds = population - blues
    M = np.zeros(N*N, dtype=np.int8)
    M[:int(reds)] = 1
    M[-int(vacant):] = -1
    np.random.shuffle(M)
    return M.reshape(N,N)

def evolve(M, boundary='wrap'):
    """
    Args:
        M (numpy.array): the matrix to be evolved
        boundary (str): Either wrap, fill, or symm
    If the similarity ratio of neighbours
    to the entire neighborhood population
    is lower than the SIM_T,
    then the individual moves to an empty house.
    """
    KERNEL = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.int8)
    kws = dict(mode='same', boundary=boundary)
    b_neighs = convolve2d(M == 0, KERNEL, **kws)
    r_neighs = convolve2d(M == 1, KERNEL, **kws)
    neighs   = convolve2d(M != -1,  KERNEL, **kws)

    b_dissatified = (b_neighs / neighs < SIM_T) & (M == 0)
    r_dissatified = (r_neighs / neighs < SIM_T) & (M == 1)
    M[r_dissatified | b_dissatified] = - 1
    vacant = (M == -1).sum()

    n_b_dissatified, n_r_dissatified = b_dissatified.sum(), r_dissatified.sum()
    filling = -np.ones(vacant, dtype=np.int8)
    filling[:n_b_dissatified] = 0
    filling[n_b_dissatified:n_b_dissatified + n_r_dissatified] = 1
    np.random.shuffle(filling)
    M[M==-1] = filling

N = 500      # Grid will be N x N
SIM_T = 0.7  # Similarity threshold (that is 1-τ)
EMPTY = 0.1  # Fraction of vacant properties
B_to_R = 0.3   # Ratio of blue to red people

p=rand_init(N, B_to_R,EMPTY)
plt.figure(1)




count1=np.count_nonzero(p==1)

ratio1=count1/(N*N)
plt.imshow(p)
timestep=100
for i in range(0,timestep):
    evolve(p)

count=np.count_nonzero(p==1)
ratio2=count/(N*N)

plt.figure(2)

plt.imshow(p)

negsx, negsy=np.where(p==-1)
negs=np. array((negsx,negsy)).T
zeros=0
ones=0
for coords in negs:
    rng=random.randint(0,9)
    
    if rng <=2:
        p[coords[0]][coords[1]]=0
        zeros=zeros+1
        
    elif rng>2:
        p[coords[0]][coords[1]]=1
        ones=ones+1

    
    
plt.figure(3)
count3=np.count_nonzero(p==1)
ratio3=count/(N*N)
plt.imshow(p)

