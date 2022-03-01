# -*- coding: utf-8 -*-
"""
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

alloy=np.random.choice(a=[False, True], size=(N, N), p=[p, 1-p])

plt.imshow(alloy, interpolation='none', cmap='viridis')


random_x_list = np.random.randint(0,100,size=N)
random_y_list = np.random.randint(0,100,size=N)


#def find_neighbours(alloy,x,y):
#    
#    

no_of_neighbours=8
tree = cKDTree(alloy)
dists = tree.query(alloy, no_of_neighbours)
nn_dist = dists[0][:, 1]
neighbourarr=np.linspace(0,no_of_neighbours,no_of_neighbours).astype(int).astype(str)

neighbour_list=pd.DataFrame(dists[1],columns=neighbourarr)

#
#for i, x in enumerate(random_x_list):
#    print(random_x_list[i],random_y_list[i])
#    

    
