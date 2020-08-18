# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:52:09 2020

@author: Clayton
"""

from functions import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import scipy as scp
from numpy import *

def mult(matrix1,matrix2):
    # Matrix multiplication
    if len(matrix1[0]) != len(matrix2):
        # Check matrix dimensions
        print('Matrices must be m*n and n*p to multiply!')
    else:
        # Multiply if correct dimensions
        new_matrix = np.zeros(len(matrix1),len(matrix2[0]))
        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    new_matrix[i][j] += matrix1[i][k]*matrix2[k][j]
        return new_matrix

directory = 'C:\\Users\\Clayton\\Downloads'
file = 'OPT_data_release.csv'

os.chdir(directory)
df=pd.read_csv(file, delimiter=',',skiprows=1)
pos=df[df['Shunt Resistor (mV)'] > 0]
firstpos=pos.iloc[0::2,:]
secondpos=pos.iloc[1::2,:]
#
#plt.scatter(df['Shunt Resistor (mV)'],df['RF (Hz)'])
#plt.scatter(pos['Shunt Resistor (mV)'],pos['RF (Hz)'])
#plt.scatter(firstpos['Shunt Resistor (mV)'],firstpos['RF (Hz)'])
#plt.scatter(secondpos['Shunt Resistor (mV)'],secondpos['RF (Hz)'])
secondpos=secondpos.iloc[1::2,:]
firstpos=firstpos.iloc[1::2,:]

#
#m1,c1=np.polyfit(firstpos['Shunt Resistor (mV)'].values,firstpos['RF (Hz)'].values,1)
#m2,c2=np.polyfit(secondpos['Shunt Resistor (mV)'].values,secondpos['RF (Hz)'].values,1)
#
#
##fig=plt.figure(2)
#
#plt.plot(firstpos['Shunt Resistor (mV)'], m1*firstpos['Shunt Resistor (mV)']+c1)
plt.scatter(firstpos['Shunt Resistor (mV)'],firstpos['RF (Hz)'])


#plt.plot(secondpos['Shunt Resistor (mV)'], m2*secondpos['Shunt Resistor (mV)']+c2)
plt.scatter(secondpos['Shunt Resistor (mV)'],secondpos['RF (Hz)'])
#
#oldx=pos['Shunt Resistor (mV)'].values
#oldy=pos['RF (Hz)'].values
#newx = (pos['Shunt Resistor (mV)'].values)*np.cos(45* np.pi / 180)
#newy = (pos['RF (Hz)'].values)*np.sin(45 * np.pi / 180)
#
#
#plt.scatter(oldx,oldy)
#plt.scatter(newx,newy)
##
#RotMatrix = np.zeros((3,3))
#RotMatrix[0][0]=cos(-45)
#RotMatrix[0][1]=-1*sin(-45)
#RotMatrix[1][0]=sin(-45)
#RotMatrix[1][1]=cos(-45)
#RotMatrix[2][2]=1
#
#rotatedx=np.dot(RotMatrix, oldx)