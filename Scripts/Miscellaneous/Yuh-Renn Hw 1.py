# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:08:07 2020

@author: Clayton
"""
import numpy as np
import matplotlib.pyplot as plt


#def f(x):
#    return (1-np.exp(x))/(1+np.exp(x)+np.exp(-x))

def f(x, a,d,c):
    return x, a*x-2*x*x*np.exp(-d*x)+c

def f_prime(x,a,d,c):
    return x, a+(2*d*x*x-4*x)*np.exp(-d*x)

a=1
d=1
c=0

x=np.arange(-25,10,0.01)
x=np.round(x,3)
y=f(x,a,d,c)[1]



dx=np.diff(x)[0]
y_prime=np.gradient(y)/dx

#counter=100
#x=0
#dx=0.01
#for i  in np.arange(counter):
#
#    y=f(x,a,d,c)[1]
#    x=f(x,a,d,c)[0]
#    y_prime=f_prime(x,a,d,c)[1]
#    x=x+y/y_prime
#    print(x)

#y_dprime=np.gradient(y_prime)/dx
plt.plot(x,y)
#plt.plot(x,y_prime)
#plt.plot(x,y_dprime)
