# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 21:35:58 2022

@author: me_hi
"""
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import random
import pandas as pd

import numpy as np

N=10
L=100
i_Ga=0
i_In=1
M = np.full(L*N*N, i_Ga) #

l=3
kernel=np.ones(shape=(l,l,l))
kernel[int(np.ceil(l/2-1)),int(np.ceil(l/2-1)),int(np.ceil(l/2-1))]=0

tot=M.size
composition=0.3
uniformity=0.8

init_In=composition*uniformity*tot

M[:int(init_In)] = i_In

np.random.shuffle(M)

M=M.reshape(L,N,N)

remainder=composition*tot-init_In

Ga=M==i_Ga


Ga_coords=np.array(np.where(Ga)).T

Gadf=pd.DataFrame(Ga_coords, columns=['x','y','z'])
kws = dict(mode='same')
Ga_neighs = convolve(M == i_Ga, kernel, **kws)
In_neighs = convolve(M == i_In, kernel, **kws)

In_neighs[In_neighs<0.00001]=0
neighs = convolve(M != 1000, kernel, **kws)

spatial_probability=In_neighs/neighs
spatial_probability[spatial_probability<0.0001]=0

test=spatial_probability/np.sum(spatial_probability)
num=test*remainder
unique,counts=np.unique(num,return_counts=True)


# for i in range(remainder):
    