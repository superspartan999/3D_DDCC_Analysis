# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:55:35 2019

@author: Clayton
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from math import floor, sqrt




def SiLENSE(FileName):
    
    file=pd.read_csv(FileName,delimiter='	',header=None, engine='python')
    headers=file.iloc[0]
    
    newfile=pd.DataFrame(file.values[1:],columns=headers)
    
    plt.plot(newfile["Bias"],newfile["Current_density"])

    return newfile
    
def OneD(FileName, legend_label):
    file=pd.read_csv(FileName,header=None)

    plt.plot(file[0],file[3],label=legend_label)
    
    
    
    return file
       
def DDCC(FileName, legend_label):
    
    file=pd.read_csv(FileName,delimiter='  ',header=None, engine='python')

    plt.plot(file[1],file[14]/(30e-7)**2,label=legend_label)
    
    
    
    return file

def silvaco(FileName):
    
    file=pd.read_csv(FileName,delimiter='	',header=None, engine='python')
    
    plt.plot(-file[5],file[3])
    
    return file
#
directory="C:\\Users\\Clayton\\Google Drive\\Research\\Simulations\\10nmAlGaN\\"
os.chdir(directory)
#x=DDCC("IV curves 10nm AlGaN - Copy.ivn", "3D Simulation")  
#y=DDCC("IV curves 30nm AlGaN - Copy.ivn", "3D Simulation")
#z=DDCC("IV curves 50nm AlGaN - Copy.ivn", "3D Simulation") 
#

directory="D:\\1"
os.chdir(directory)
file=pd.read_csv(filename,header=None)
x=OneD("10nmIV.csv",'1D Simulation')
y=OneD("30nmIV.csv",'1D Simulation')
z=OneD("50nmIV.csv",'1D Simulation')

plt.grid(color='black')
plt.xlim(-4,0.5)
plt.ylim(-5000,5000)
plt.legend()
#y=SiLENSE("IV_p_struc_10.dat")

#z=silvaco("IV_silvaco_10nm.txt")
#
#plt.scatter(file[1],file[14]/(30e-7)**2)