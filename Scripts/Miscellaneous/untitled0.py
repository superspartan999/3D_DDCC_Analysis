# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:08:07 2020

@author: Clayton
"""
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (1-np.exp(x))/(1+np.exp(x)+np.exp(-x))


x=np.arange(-25,25,0.01)
y=f(x)
dx=np.diff(x)[0]
y_prime=np.gradient(y)/dx
y_dprime=np.gradient(y_prime)/dx
plt.plot(x,y)
plt.plot(x,y_prime)
plt.plot(x,y_dprime)
