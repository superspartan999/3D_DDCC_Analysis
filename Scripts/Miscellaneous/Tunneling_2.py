#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:56:27 2019

@author: DEAGUEROS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy as scp
#from scipy.constants import hbar
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import quad
from scipy.interpolate import interp1d

directory="/Users/DEAGUEROS/Desktop/DDCC/"
directory="D:/Box Files/Box Sync/Evelyn Data"
directory="C:/Users/Clayton/Downloads/ddcc"
directory='D:/1D-DDCC'
#directory="C:/Users/Clayton/Downloads/test-DDCC"
#directory="D:/transport structure project/TJ Calculation"
#directory="D:/Box Files/Box Sync/Evelyn Data/InGaN"
os.chdir(directory)
#
#calculate exponent for WKB tunneling
#def k_z(Ez,m,Ec,carrier):
#    
#    if carrier=='electron':
#        return np.sqrt((2*m/hbar**2.0)*(Ec-Ez))
#    
#    if carrier=='hole':
#        return np.sqrt((2*m/hbar**2.0)*(Ez-Ec))
    
m0=0.511*10**6/(3.0e8)**2.0
hbar=6.582119569e-16
k=8.617e-5	
m_n=0.13*m0
T=300
e=-1.6e-19

#def BanDpl:
thickness=32
thicknessarray=[13]
bgap=np.empty(len(thicknessarray))
#for i,thickness in enumerate(thicknessarray):
import os


temp=[]
keyword = '22'
for fname in os.listdir(directory):
    if keyword in fname:
        if 'cb.res' in fname:
            if 'vg_-' in fname:

                temp.append(fname)

#
filename=temp[0]
dat=pd.read_csv(filename, delimiter=r"\s+", engine='python',header=None)
labels=['x', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
                 'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
                 'uEv','uEv2','effective trap','stimulate R']
dat.columns = labels

#dat['x']=dat['x'].apply(lambda x: x*10e-2)
charge=dat['Electric field'].diff()/dat['x'].diff()
diffcharge=charge.diff()/dat['x'].diff()
#dat=dat.rename(columns={'Efp':'Hole Fermi Level'})
dat=dat.rename(columns={'Ec':'Conduction Band'})
dat=dat.rename(columns={'Ev':'Valence Band'})
#calculate depletion width


#find turning point in band diagram by analysing peaks on the derivative of charge
(peak_index,info)=find_peaks(-diffcharge,prominence=10000000)
depletion_width=dat['x'].iloc[peak_index[1]]-dat['x'].iloc[peak_index[0]]
a=dat['x'].iloc[peak_index[0]]
b=dat['x'].iloc[peak_index[1]]


#create dataframe of Ec potential
Ec=dat[['x','Conduction Band']].copy()

Ec['x']=Ec['x'].values[::-1]
Ec=Ec.sort_values(['x','Conduction Band'],ascending=[True,True]).reset_index(drop=True)
#Ec=Ec.iloc[peak_index[0]:peak_index[1]+1].reset_index(drop=False)

Ec['x']=Ec['x']-Ec['x'].iloc[0]

#interpolate values in potential
f = interp1d(Ec['x'].values, Ec['Conduction Band'].values, kind='cubic')
x=Ec['x'].values
x_new=np.linspace(Ec['x'].iloc[0],Ec['x'].iloc[-1],10000)

plt.plot(x,Ec['Conduction Band'].values)
plt.plot(x_new,f(Ec['x'].values))
#filelist=np.array(temp)
#names=np.linspace(0.2,5,25)
#J=np.zeros(filelist.size)
#for i, name in enumerate(filelist):
#    filename=name
#    #filename='40_AlGaN_UID_2_result.out.vg_-0.000-cb.res'
#    #filename='13nm_0.14dopedAlGaN_result.out.vg_-0.00-cb.res'
#    #filename='40nm_uidGaN_result.out.vg_-0.00-cb.res'
#    #filename='AlGaN_0.6_30nm_result.out.vg_-0.00-cb.res'
#    #filename='A3_10nmAlGN_result.out.vg_0.000-cb.res'
#    
#    dat=pd.read_csv(filename, delimiter=r"\s+", engine='python',header=None)
#    #dat=dat[(dat['x']>3.0e-6) & (dat['x']<(3.0e-6+thickness))].reset_index(drop=True)
#    labels=['x', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
#                 'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
#                 'uEv','uEv2','effective trap','stimulate R']
#    dat.columns = labels
#    #dat['X']=dat['x'].apply(lambda x: x*10e6)
#    
#    #    dat=dat[(dat['x']>5.0e-6) & (dat['x']<(5.0e-6+(thickness+1)*1e-7))].reset_index(drop=True)
#    dat['X']=dat['x'].apply(lambda x: x*10e6)
#    dat['x']=dat['x'].apply(lambda x: x*10e-2)
#    dat['bandgap']=dat['Ec']-dat['Ev']
#    dat['bandgap'].mean()
#    charge=dat['Electric field'].diff()/dat['X'].diff()
#    diffcharge=charge.diff()/dat['X'].diff()
#    #dat=dat.rename(columns={'Efp':'Hole Fermi Level'})
#    dat=dat.rename(columns={'Ec':'Conduction Band'})
#    dat=dat.rename(columns={'Ev':'Valence Band'})
#    #calculate depletion width
#    
#    #find turning point in band diagram by analysing peaks on the derivative of charge
#    (peak_index,info)=find_peaks(-diffcharge,prominence=1000000)
#    depletion_width=dat['x'].iloc[peak_index[1]]-dat['x'].iloc[peak_index[0]]
#    a=dat['x'].iloc[peak_index[0]]
#    b=dat['x'].iloc[peak_index[1]]
#    
#    
#    #create dataframe of Ec potential
#    Ec=dat[['x','Conduction Band']].copy()
#    
#    Ec['x']=Ec['x'].values[::-1]
#    Ec=Ec.sort_values(['x','Conduction Band'],ascending=[True,True]).reset_index(drop=True)
#    Ec=Ec.iloc[peak_index[0]:peak_index[1]+1].reset_index(drop=False)
#    
#    Ec['x']=Ec['x']-Ec['x'].iloc[0]
#    
#    #interpolate values in potential
#    f = interp1d(Ec['x'].values, Ec['Conduction Band'].values, kind='cubic')
#    
#    
#    def T(Ez):
#        
#        x_new = np.linspace(Ec['x'].iloc[0], Ec['x'].iloc[-1], 10001)
#        
#        x_new_pd=pd.DataFrame(x_new)
#        x_new_pd[1]=f(x_new_pd[0])-Ez
#        x_new_pos=x_new_pd.loc[x_new_pd[1]>=0]
#        
#    
#        a1=x_new_pos[0].iloc[0]
#        b1=x_new_pos[0].iloc[-1] 
#    
#    
#        I1=quad(lambda x :np.sqrt((2*m_n/hbar**2.0)*(f(x)-Ez)), a1, b1)[0]
#        return np.exp(-2*(I1)) 
#    
#    Efp=dat['Efp'].iloc[-2]
#    Efn=dat['Efn'].iloc[1]
#    
#    m=9.11e-31
#    kb=1.38e-23
#    Temp=300
#    h=1.09e-34
#    
#    #tunneling current prefactor
#    A=e*m_n*k*Temp/(2*np.pi**2*hbar**3)
#    
#    I=lambda Ez: T(Ez)*np.log((1+np.exp(Efp-Ez)/k*300)/(1+np.exp(Efn-Ez)/k*300))
#    
#    
#    J[i]=quad(lambda Ez: A*T(Ez)*np.log((1+np.exp(Efp-Ez)/k*300)/(1+np.exp(Efn-Ez)/k*300)), 0.1, Ec['Conduction Band'].iloc[-1])[0]
#J2=quad(lambda Ez: A*Tr(Ez)*np.log((1+np.exp(Efp-Ez)/k*300)/(1+np.exp(Efn-Ez)/k*300)), Ec['Conduction Band'].max(), np.inf )



#function to define Tunneling probability
#def T(Ez):
#    
#    x_new = np.linspace(Ec['x'].iloc[0], Ec['x'].iloc[-1], 10001)
#    
#    x_new_pd=pd.DataFrame(x_new)
#    x_new_pd[1]=f(x_new_pd[0])-Ez
#    
#    x_new_pos=x_new_pd.loc[x_new_pd[1]>0]
#    x_new_neg=x_new_pd.loc[x_new_pd[1]<0]
#    
#    if x_new_pos.size==0:
#        
#        a2=x_new_neg[0].iloc[0]
#        b2=x_new_neg[0].iloc[-1]
#            
#        I2=quad(lambda x :np.sqrt((2*m_n/hbar**2.0)*(Ez-f(x))), a2, b2)[0]
#        return I2
#        
#    elif x_new_neg.size==0:
#        
#        a1=x_new_pos[0].iloc[0]
#        b1=x_new_pos[0].iloc[-1] 
#    
#    
#        I1=quad(lambda x :np.sqrt((2*m_n/hbar**2.0)*(f(x)-Ez)), a1, b1)[0]
#        return I1
#
#    else:
#        #shifting limits of integral
#        a1=x_new_pos[0].iloc[0]
#        b1=x_new_pos[0].iloc[-1] 
#        a2=x_new_neg[0].iloc[0]
#        b2=x_new_neg[0].iloc[-1]
#        
#    
#        I1=quad(lambda x :np.sqrt((2*m_n/hbar**2.0)*(f(x)-Ez)), a1, b1)[0]
#        I2=quad(lambda x :np.sqrt((2*m_n/hbar**2.0)*(Ez-f(x))), a2, b2)[0]
#        return np.exp(-2*(I1+I2)) 
    #function to define Tunneling probability
#J=quad(T,0,1)
#dat['X']=dat['x']

#dat.plot(x="X", y=["Ev","Ec"])
#
#plt.plot(dat['X'],dat['Electric field'],label='Electric Field')
##plt.plot(dat['X'],dat['p'],label='charge')
#
###plt.plot(dat['X'],dat['p'],label='p')
##plt.plot(dat['X'],dat['active dopant'],label='Na')
#plt.plot(dat['X'],dat['Conduction Band'],label='Conduction Band')
#plt.plot(dat['X'],dat['Valence Band'],label='Valence Band')
##plt.plot(dat['X'],dat['Hole Fermi Level'],label='Hole Fermi Level')
#
#
##plt.plot( dat['Ev'],dat['Ec'])
##bgap[i]=dat['bandgap'].mean()
#
#plt.xlabel('z(nm)', fontsize=12)
#
#plt.ylabel('Band', fontsize=12)

#dat.to_csv(filename+'.csv')

#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))



#plt.suptitle('50nmAlGaN (undoped)', fontsize=20)

#plt.savefig(filename + '.pdf')