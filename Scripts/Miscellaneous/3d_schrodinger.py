# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 17:47:08 2022

@author: me_hi
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
# plt.style.use(['science','notebook'])
from scipy import sparse
import open3d as o3d
from mayavi import mlab

N=50
X, Y, Z = np. meshgrid(np.linspace(0,1,N, dtype=float),
                    np.linspace(0,1,N, dtype=float),
                    np.linspace(0,1,N, dtype=float))


def get_potential(x,y,z):
    return 0*x
# def get_potential(x, y):
#     return np.exp(-(x-0.3)**2/(2*0.1**2))*np.exp(-(y-0.3)**2/(2*0.1**2))

V = get_potential(X,Y,Z)
V=np.random.choice(a=[0,10],size=(N,N,N))
diag = np.ones([N])
diags = np.array([diag, -2*diag, diag])
D = sparse.spdiags(diags, np.array([-1,0,1]), N, N)
T = -1/2 * sparse.kronsum(sparse.kronsum(D,D),D)
U = sparse.diags(V.reshape(N**3), (0))
H = T+U

eigenvalues, eigenvectors = eigsh(H, k=100, which='SM')


def get_e(n):
    return eigenvectors.T[n].reshape((N,N,N))
# plt.figure(figsize=(9,9))

# fig=plt.figure()
# ax = fig.add_subplot(projection='3d')

probability=get_e(0)**2
figure= mlab.figure('DensityPlot')
pts = mlab.points3d(X, Y, Z, probability)
# grid = mlab.pipeline.scalar_field(X, Y, Z, probability)
# min = probability.min()
# max=probability.max()
# mlab.pipeline.volume(grid, vmin=min, vmax=min + .5*(max-min))
# mlab.axes()
# mlab.show()
# filter_val=5e-5
# probability[probability<filter_val]=0
# probability[probability>filter_val]=1
# Z[probability<1]=np.nan
# probability[probability<1]=np.nan


# ax.scatter(X,Y,Z)
# o3d.visualization.draw_geometries([probability])
