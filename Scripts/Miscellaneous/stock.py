# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:08:07 2020

@author: Clayton
"""
import numpy as np
import matplotlib.pyplot as plt
import OriginExt as O


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

x=np.random(100)
y=np.random(100)
stadev=np.zeros(100)

for i in range(100):
    stadev[i]=np.std(x,y)