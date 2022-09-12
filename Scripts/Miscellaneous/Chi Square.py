# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 23:06:50 2022

@author: me_hi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
n=1000
#x-axis ranges from 0 to 20 with .001 steps
x = np.arange(0, n, 0.001)
colors=plt.cm.viridis(np.linspace(0,1,n))
#plot Chi-square distribution with 4 degrees of freedom
for i in range(n):
    plt.plot(x, chi2.pdf(x, df=i+5),color=colors[i])
    plt.xlim(0,100)
