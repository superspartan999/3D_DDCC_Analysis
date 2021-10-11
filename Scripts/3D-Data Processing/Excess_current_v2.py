# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 23:55:08 2021

@author: me_hi
"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import re

from scipy.interpolate import interp1d
import os

directory='G:\My Drive\Research\Transport Structure 2020\\072120AB - 30nm InGaN'
os.chdir(directory)

def interpolator(datt):
    datt=datt.dropna()
    x=datt['V'].values
    y=datt['I'].values
    # plt.plot(x,y)
    
    f=interp1d(x,y)
    xnew=np.linspace(x[0],x[-1],200)
    
    # plt.plot(xnew, f(xnew))
    
    return f, xnew
filelist=[]
for fname in os.listdir(directory):
            if '.csv' in fname:
            
               temp=fname
               temp=temp.replace('umr.csv','')
               filelist.append(temp)
               
# for file in filelist:
#     file = file.replace("umr.csv", " ")
filelist=[eval(x) for x in filelist]

filelist=sorted(filelist)

filelist=[str(x) for x in filelist]               

DataFrameDict = {elem : pd.DataFrame for elem in filelist}

list_IV=[]

# master_IV=pd.DataFrame()


for key in DataFrameDict.keys():
    DataFrameDict[key] = pd.read_csv(key+'umr.csv')
    temp = re.findall(r'\d+', key)
    diameter=[float(x) for x in temp]
    area=(np.pi*(diameter[0]*10**-4)**2/4)
    
    DataFrameDict[key]['I']=DataFrameDict[key]['I']/area
    x=DataFrameDict[key]['V']
    y=abs(DataFrameDict[key]['I'])
    plt.plot(x,y,label=key)
    list_IV.append(DataFrameDict[key])
    # ax=plt.axes()
    # ax.tick_params(axis='x', which='minor', bottom=False)
    # plt.yscale('log')
    # plt.grid(b=True, which='major', color='k', linestyle='-')
    # plt.grid(b=True, which='minor', color='r', linestyle='--')
    # plt.minorticks_on()
    # plt.show()


fig=plt.figure(2)
ref,x_ref=interpolator(DataFrameDict['200'])

# s150,x150=interpolator((DataFrameDict['150umr.csv']))

# v_drop_150=s150(x150)-ref(x_ref)


# ref,x_ref=interpolator(DataFrameDict['200umr.csv'])

# s175,x175=interpolator((DataFrameDict['175umr.csv']))

# v_drop_175=s175(x175)-ref(x_ref)


# plt.plot(x150,s150(x150))
# plt.plot(x_ref,ref(x_ref))
# plt.plot(x_ref,v_drop_150)
# plt.plot(x_ref,v_drop_175)

DropVDict= {elem : pd.DataFrame for elem in filelist}    

master_dropV=pd.DataFrame()

list_dropV=[]

for key in DataFrameDict.keys():
    # interpolator=DataFrameDict[key]
    temp = re.findall(r'\d+', key)
    diameter=[float(x) for x in temp]
    area=(np.pi*(diameter[0]*10**-4)**2/4)
    f,x_new=interpolator(DataFrameDict[key])
    excessI=(f(x_new)-ref(x_ref))
    DropVDict[key]=pd.DataFrame({'Voltage' : x_new,key:excessI})
    
    # plt.minorticks_on()
    plt.semilogy(DropVDict[key]['Voltage'],abs(DropVDict[key][key]),label=key)
    list_dropV.append(DropVDict[key])
    # plt.minorticks_on()

    # # plt.axes().yaxis.set_minor_locator(ml)
    # # plt.yscale('log')
    # # plt.grid(True, which="both", ls="--")
    # plt.grid(True, which="major", ls="-")
    # plt.grid(True, which="minor", ls="--")
master_dropV=pd.concat(list_dropV,axis=1)

master_IV=pd.concat(list_IV,axis=1)


master_dropV.to_csv('G:\\My Drive\\Research\\Transport Structure 2020\\072120AB - 30nm InGaN\\master\\master.csv')
master_IV.to_csv('G:\\My Drive\\Research\\Transport Structure 2020\\072120AB - 30nm InGaN\\master\\IV.csv')

# for key in DataFrameDict.keys():
    
    