# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 02:09:42 2020

@author: Clayton
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy as scp
#from scipy.constants import hbar
from scipy.signal import find_peaks, savgol_filter,peak_prominences
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA



directory='D:/1D-DDCC'
os.chdir(directory)
keyword_array=['11']

def extractDepletion(dat):


    maxEIndex = dat['Electric field'].idxmax()
    maxE = dat['Electric field'].max()

    Filter = (dat['Electric field'] > maxE * 0.5) & (dat.index <= maxEIndex)
    anaDat_1 = dat['Electric field'][Filter]
    m, b, r, p, stderr = linregress(x=anaDat_1.index, y=anaDat_1)
    w1 = -(b/m)

    Filter = (dat['Electric field'] > maxE * 0.5) & (dat.index >= maxEIndex)
    anaDat_2 = dat['Electric field'][Filter]
    if len(anaDat_2) > 1:
        m, b, r, p, stderr = linregress(x=anaDat_2.index, y=anaDat_2)
        w2 = -(b/m)
    else:
        w2 = 0



    return dat['x'].iloc[int(w1)],dat['x'].iloc[int(w2)],int(w1),int(w2)

#for keyword in keyword_array:
#    #make a temp array of list of files within directory
#    temp=[]
#
#    
#    #search for files containing specific keywords ie. 'cb.res' and 'vg_'
#    for fname in os.listdir(directory):
#        if keyword in fname:
#            if 'cb.res' in fname:
#                if 'vg_-' in fname:
#                    if '-5.000' in fname:
#    
#                        temp.append(fname) 
#                        
#                        
#    J=np.zeros(len(temp))
#    for i, name in enumerate(temp):
    
scaler=np.linspace(1,2,10)
#for scale in scaler:
filename='TJ_11_Diagram_result.out.vg_-1.000-cb.res'
        #filename='40_AlGaN_UID_2_result.out.vg_-0.000-cb.res'
        #filename='13nm_0.14dopedAlGaN_result.out.vg_-0.00-cb.res'
        #filename='40nm_uidGaN_result.out.vg_-0.00-cb.res'
        #filename='AlGaN_0.6_30nm_result.out.vg_-0.00-cb.res'
        #filename='A3_10nmAlGN_result.out.vg_0.000-cb.res'

dat=pd.read_csv(filename, delimiter=r"\s+", engine='python',header=None)
#dat=dat[(dat['x']>3.0e-6) & (dat['x']<(3.0e-6+thickness))].reset_index(drop=True)
labels=['x', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
             'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
             'uEv','uEv2','effective trap','stimulate R']
dat.columns = labels
#dat['X']=dat['x'].apply(lambda x: x*10e6)
dat['Electric field']=-dat['Electric field']

#    dat=dat[(dat['x']>5.0e-6) & (dat['x']<(5.0e-6+(thickness+1)*1e-7))].reset_index(drop=True)
dat['X']=dat['x'].apply(lambda x: x*10e6)
dat['x']=dat['x'].apply(lambda x: x*10e-2)
#    dat['bandgap']=dat['Ec']-dat['Ev']
#    dat['bandgap'].mean()
#    charge=dat['Electric field'].diff()/dat['X'].diff()
#    diffcharge=charge.diff()/dat['X'].diff()
#dat=dat.rename(columns={'Efp':'Hole Fermi Level'})
dat=dat.rename(columns={'Ec':'Conduction Band'})
dat=dat.rename(columns={'Ev':'Valence Band'})
#    #calculate depletion width

a,b, a_index,b_index=extractDepletion(dat)
#    a,b, a_index,b_index=find_depletion_width(dat)

#create dataframe of Ec potential
Ec=dat[['x','Conduction Band']].copy()
Ev=dat[['x','Valence Band']].copy()

Ec['x']=Ec['x'].values[::-1]
Ec=Ec.sort_values(['x','Conduction Band'],ascending=[True,True]).reset_index(drop=True)
Ec=Ec.iloc[a_index:b_index].reset_index(drop=False)
Ec['x']=Ec['x']-Ec['x'].iloc[0]


Ev['x']=Ev['x'].values[::-1]
Ev=Ev.sort_values(['x','Valence Band'],ascending=[True,True]).reset_index(drop=True)
Ev=Ev.iloc[a_index:b_index].reset_index(drop=False)
Ev['x']=Ev['x']-Ev['x'].iloc[0]


#Ec['Conduction Band']=scale*Ec['Conduction Band']
#    Ec['x']=scale*Ec['x']
x_new=np.linspace(Ec['x'].iloc[0],Ec['x'].iloc[-1],100000)
#interpolate values in potential
coeff=np.polyfit(Ec['x'].values, Ec['Conduction Band'].values ,3)
f=np.poly1d(coeff)

my_func=lambda x:coeff[0]*x**3+coeff[1]*x**2+coeff[2]*x+coeff[3]
#my_func=lambda x:(-1.2e22)*x**3+(1.88497e15)*x**2+(-2e7)*x+0.11936

#plt.plot(Ec['x'],Ec['Conduction Band'])
#plt.plot(x_new,f(x_new))
plt.plot(x_new,my_func(x_new))