# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 12:41:41 2021

@author: Clayton
"""


import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
from scipy import sparse
from numpy import random
import scipy.stats as st
from scipy.ndimage import gaussian_filter

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

def get_potential(x,y):
    return 0*x

def get_e(n):
    return eigenvectors.T[n].reshape((N,N))


sigma_y = 1.0
sigma_x = 1.0
sigma=[sigma_y,sigma_x]
sigma=3
N = 150
X, Y = np.meshgrid(np.linspace(0,1,N, dtype=float),
                   np.linspace(0,1,N, dtype=float))

V = get_potential(X,Y)
p=0.03
V=random.choice(a=[1, 0], size=(N, N), p=[p, 1-p])


V_blur = gaussian_filter(V,1)
plt.figure(1)

plt.imshow(V)

plt.figure(2)

plt.imshow(V_blur)
#diag = np.ones([N])
#diags = np.array([diag, -2*diag, diag])
#D = sparse.spdiags(diags, np.array([-1,0,1]), N, N)
#T = -1/2 * sparse.kronsum(D,D)
#U = sparse.diags(V.reshape(N**2), (0))
#H = T+U
#
#
#eigenvalues, eigenvectors = eigsh(H, k=100, which='SM')
#
#plt.figure(figsize=(9,9))
#plt.contourf(X, Y, get_e(99)**2, 20)
#
#alpha = eigenvalues[0]/2
#E_div_alpha = eigenvalues/alpha
#_ = np.arange(0, len(eigenvalues), 1)
#plt.scatter(_, E_div_alpha)