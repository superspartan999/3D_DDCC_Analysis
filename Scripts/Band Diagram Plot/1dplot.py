# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:13:58 2019

@author: Clayton
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as scp
import numpy as np



directory="E:\\Google Drive\\Research\\Transport Structure Project\\TJ Calculation"



directory="D:\\1D-DDCC"

os.chdir(directory)
colnames=['x', 'Ec', 'Ev', 'Efn','Efp','n','p','Jn','Jp','Rad','nonRad','Auger', 'RspPL', 'eb', 'ebh' 'gen', 'active dopants','impurity'
                                                                           '1/u_Ec', '1/u_Ev', '1/u_Ehh','Electric field', 'mu_n','mu_p','uEc','uEv',
                                                                           'uEv2','effective traps','layernum']
colnames={ 0:'x', 1:'Ec', 2:'Ev', 3:'Efn',4:'Efp',5:'n',6:'p',7:'Jn',8:'Jp',9:'Rad'
          ,10:'nonRad',11:'Auger', 12:'RspPL', 13:'eb', 14:'ebh',15:'gen', 16:'active dopants'
          ,17:'impurity',18:'1/u_Ec', 19:'1/u_Ev', 20:'1/u_Ehh',21:'Electric field', 22:'mu_n',23:'mu_p',24:'uEc'
          ,25:'uEv',26:'uEv2',27:'effective traps',28:'layernum'}



df=pd.read_csv('TJ_54_result.out.vg_-1.00-cb.res', delimiter='   ',index_col=False, names=colnames,engine='python')


df=pd.read_csv('TJ_11_result.out.vg_-6.00-cb.res', delimiter='   ',index_col=False, names=colnames,engine='python')

df1=pd.read_csv('TJ_54_result.out.vg_0.00-cb.res', delimiter='   ',engine='python', header=None)
df=df.rename(columns=colnames)

df['dE'] = df['Electric field'] - df['Electric field'].shift(-1)
df['d2E'] = df['dE'] - df['dE'].shift(-1)
df['d3E'] =df['d2E'] - df['d2E'].shift(-1)

df['x']=df['x'].div(1e-7)

plt.plot(df['x'],df['d2E'])

x=np.array(df['x'])

n=100

extremas = scp.argrelextrema(df['d2E'].values,np.less_equal, order=n
                             )

df['min'] = df.iloc[scp.argrelextrema(df['d2E'].values, np.less_equal, order=n)[0]]['d2E']

depletion=df['x'].iloc[extremas[0][2]]-df['x'].iloc[extremas[0][1]]

depletion=df['x'].iloc[extremas[0][3]]-df['x'].iloc[extremas[0][0]]


plt.plot(df['x'],df['d2E'])

df=pd.read_csv('TJ_11_result.out.vg_-6.00-cb.res', delimiter='   ',index_col=False, names=colnames,engine='python')
df1=pd.read_csv('TJ_54_result.out.vg_0.00-cb.res', delimiter='   ',engine='python', header=None)
df=df.rename(columns=colnames)

df['dE'] = df['Electric field'] - df['Electric field'].shift(-1)
df['d2E'] = df['dE'] - df['dE'].shift(-1)
df['d3E'] =df['d2E'] - df['d2E'].shift(-1)

df['x']=df['x'].div(1e-7)

plt.plot(df['x'],df['d2E'])

x=np.array(df['x'])

n=100

extremas = scp.argrelextrema(df['d2E'].values,np.less_equal, order=n
                             )

df['min'] = df.iloc[scp.argrelextrema(df['d2E'].values, np.less_equal, order=n)[0]]['d2E']

depletion=df['x'].iloc[extremas[0][3]]-df['x'].iloc[extremas[0][0]]

plt.plot(df['x'],df['d2E'])


df=pd.read_csv('TJ_11_result.out.vg_-6.00-cb.res', delimiter='   ',index_col=False, names=colnames,engine='python')
df1=pd.read_csv('TJ_54_result.out.vg_0.00-cb.res', delimiter='   ',engine='python', header=None)
df=df.rename(columns=colnames)

df['dE'] = df['Electric field'] - df['Electric field'].shift(-1)
df['d2E'] = df['dE'] - df['dE'].shift(-1)
df['d3E'] =df['d2E'] - df['d2E'].shift(-1)

df['x']=df['x'].div(1e-7)

plt.plot(df['x'],df['d2E'])

x=np.array(df['x'])

n=100

extremas = scp.argrelextrema(df['d2E'].values,np.less_equal, order=n
                             )

df['min'] = df.iloc[scp.argrelextrema(df['d2E'].values, np.less_equal, order=n)[0]]['d2E']

depletion=df['x'].iloc[extremas[0][3]]-df['x'].iloc[extremas[0][0]]

plt.plot(df['x'],df['d2E'])

plt.scatter(df['x'], df['min'], c='r')
