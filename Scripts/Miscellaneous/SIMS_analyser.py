# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:07:44 2020

@author: Clayton
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy as scp

directory="D:\\Research\\SIMS\\Indium Surfactant"

filelist=[]
for fname in os.listdir(directory):
            if '.csv' in fname:
               filelist.append(fname)
                
#       
#               
#plt.figure(5)
filename=filelist[1]

os.chdir(directory)
dat=pd.read_csv(filename, delimiter=",", engine='python',header=None)
dat[0].iloc[0]=dat[0].iloc[0][3:]
df=dat.copy()


elements=df.loc[0].dropna()
elements=sorted(elements)

m=np.linspace(0,len(elements)-1,len(elements))

X=3*m

for entry in X: 
    df[entry+1].loc[0]=df[entry].loc[0]
    df[entry+2].loc[0]=df[entry].loc[0]
    
    
df.columns=df.loc[0]
df=df[1:]
#create unique list of names
UniqueNames = df.columns.unique()

#create a data frame dictionary to store your data frames
DataFrameDict = {elem : pd.DataFrame for elem in UniqueNames}
plt.figure()
for key in DataFrameDict.keys():
    DataFrameDict[key] = df[key]
    DataFrameDict[key].columns=df[key].iloc[0]
    DataFrameDict[key]=DataFrameDict[key][1:]
    DataFrameDict[key]=DataFrameDict[key].drop(columns=['Time'])
    
    
elements=sorted(elements)
#elements=['28Si']

for key in elements:
    DataFrameDict[key]['Depth[nm]']=DataFrameDict[key]['Depth[nm]'].dropna()
    DataFrameDict[key]['C[atom/cm3]']=DataFrameDict[key]['C[atom/cm3]'].dropna()
    DataFrameDict[key]=DataFrameDict[key].dropna()
    x=DataFrameDict[key]['Depth[nm]'].values
    x=x.astype(np.float)
    y=DataFrameDict[key]['C[atom/cm3]'].values
    y=y.astype(np.float)
    if key =='l':
        y=y*1e-4
#    if key=='133Cs 24Mg':
#        print(y.max())
    plt.semilogy(x,y,label=key)
  
plt.legend()
plt.title(filename)
#plt.xlim([0,1700])
plt.ylim([1e15,1e20])
plt.xlabel('Depth (nm)')
plt.ylabel('Concentration (atom/cm3)')
plt.grid()

filename=filelist[0]

os.chdir(directory)
dat=pd.read_csv(filename, delimiter=",", engine='python',header=None)
dat[0].iloc[0]=dat[0].iloc[0][3:]
df=dat.copy()


elements=df.loc[0].dropna()
elements=sorted(elements)

m=np.linspace(0,len(elements)-1,len(elements))

X=3*m

for entry in X: 
    df[entry+1].loc[0]=df[entry].loc[0]
    df[entry+2].loc[0]=df[entry].loc[0]
    
    
df.columns=df.loc[0]
df=df[1:]
#create unique list of names
UniqueNames = df.columns.unique()

#create a data frame dictionary to store your data frames
DataFrameDict = {elem : pd.DataFrame for elem in UniqueNames}
plt.figure()
for key in DataFrameDict.keys():
    DataFrameDict[key] = df[key]
    DataFrameDict[key].columns=df[key].iloc[0]
    DataFrameDict[key]=DataFrameDict[key][1:]
    DataFrameDict[key]=DataFrameDict[key].drop(columns=['Time'])
    
    
elements=sorted(elements)
#elements=['28Si']

for key in elements:
    DataFrameDict[key]['Depth[nm]']=DataFrameDict[key]['Depth[nm]'].dropna()
    DataFrameDict[key]['C[atom/cm3]']=DataFrameDict[key]['C[atom/cm3]'].dropna()
    DataFrameDict[key]=DataFrameDict[key].dropna()
    x=DataFrameDict[key]['Depth[nm]'].values
    x=x.astype(np.float)
    y=DataFrameDict[key]['C[atom/cm3]'].values
    y=y.astype(np.float)
    if key =='l':
        y=y*1e-4
#    if key=='133Cs 24Mg':
#        print(y.max())
    plt.semilogy(x,y,label=key)
  
plt.legend()
plt.title(filename)
#plt.xlim([0,1700])
plt.ylim([1e15,1e20])
plt.xlabel('Depth (nm)')
plt.ylabel('Concentration (atom/cm3)')
plt.grid()



