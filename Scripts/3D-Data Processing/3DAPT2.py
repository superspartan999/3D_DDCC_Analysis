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
L=1000
i_Ga=0
i_In=1
M = np.full(L*N*N, i_Ga) #

l=4
kernel=np.ones(shape=(l,l,l))
kernel[int(np.ceil(l/2-1)),int(np.ceil(l/2-1)),int(np.ceil(l/2-1))]=0

tot=M.size
composition=0.2
uniformity=0.8

init_In=int(composition*uniformity*tot)

M[:int(init_In)] = i_In

np.random.shuffle(M)

M=M.reshape(L,N,N)

remainder=int(composition*tot-init_In)

# Ga=M==i_Ga


# Ga_coords=np.array(np.where(M==i_Ga)).T
# In_coords=np.array(np.where(M==i_In)).T    

# Gadf=pd.DataFrame(Ga_coords, columns=['x','y','z'])
kws = dict(mode='same')
Ga_neighs = convolve(M == i_Ga, kernel, **kws)
In_neighs = convolve(M == i_In, kernel, **kws)
Ga_neighs[Ga_neighs<0.00001]=0
In_neighs[In_neighs<0.00001]=0
neighs = convolve(M != 1000, kernel, **kws)
spatial_probability=In_neighs/neighs
spatial_probability[spatial_probability<0.0001]=0

isGa=(M==i_Ga)
Ga_coords=np.array(np.where(M==i_Ga)).T
Gadf=pd.DataFrame(Ga_coords, columns=['x','y','z'])
spatial_list=np.zeros(shape=len(Gadf.index))

for i, Ga_coord in enumerate(Ga_coords):
     spatial_list[i]=spatial_probability[Ga_coord[0],Ga_coord[1],Ga_coord[2]]


# Gadf['weights']=spatial_list
remainder_list=np.array(np.random.choice(Ga_coords,p=spatial_list/np.sum(spatial_list),k=int(remainder)))
 
i=0
j=0
for switch in remainder_list:
    if M[switch[0],switch[1],switch[2]]==0:
        # M[switch[0],switch[1],switch[2]]=1
        print(i)
        i+=1
    else:
        print(j)
        j+=1
    
# In_coords=np.array(np.where(M==i_In)).T    

# Indf=pd.DataFrame(In_coords, columns=['x','y','z'])

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# pos = np.where(M==i_In)
# ax.scatter(pos[0], pos[1], pos[2], c='black')
# plt.show()