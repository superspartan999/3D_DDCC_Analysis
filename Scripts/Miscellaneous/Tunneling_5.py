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
from scipy.signal import find_peaks, savgol_filter,peak_prominences
from scipy.integrate import quad
from scipy.interpolate import interp1d

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

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

m=9.11e-31
kb=1.38e-23
Temp=300
h=1.09e-34

m_n=0.20*m0
m_p=1.3*m0
T=300
e=-1.6e-19




#def BanDpl:
thickness=32
thicknessarray=[13]
bgap=np.empty(len(thicknessarray))
#for i,thickness in enumerate(thicknessarray):
import os


##
#filename=temp[0]
#dat=pd.read_csv(filename, delimiter=r"\s+", engine='python',header=None)
#labels=['x', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
#                 'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
#                 'uEv','uEv2','effective trap','stimulate R']
#dat.columns = labels

#function to find nearest array point to specific value
def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx],idx
    
    
#look for depletion width in structure. this is done by calculating the double differential of the electric field and finding the peak
def find_depletion_width(dat):

    #create new x-array for interpolation
    x_new=np.linspace(dat['x'].iloc[0],dat['x'].iloc[-1],1000)
    

    #interpolate electric field
    electric_field_interpolated = interp1d(dat['x'].values, dat['Electric field'].values, kind='cubic')

    #calculate charge from electric field gradient
    charge=np.gradient(electric_field_interpolated(x_new))/np.gradient(x_new)

    #calculate charge gradient
    diff_charge=np.gradient(charge)/np.gradient(x_new)

    #find midpoint of junction
    mid_x_ind=np.argsort(x_new)[len(x_new)//2]
    
    #find peaks in the charge gradient which correspond to the turning point in band diagram
    (peak_index,info)=find_peaks(-diff_charge,prominence=1000, width=10)
    
    
    #calculate distance of peak from center
    peak_distance=x_new[peak_index]-x_new[mid_x_ind]
    
    #find peaks closest to the center
    peak_distance=abs(x_new[peak_index]-x_new[mid_x_ind])
    window= np.argpartition(peak_distance,1)
    
    a=x_new[peak_index[window[0]]]
    b=x_new[peak_index[window[1]]]
    
    return a,b, peak_index[window[0]],peak_index[window[1]]

        
def T_n(Ez):
    
    x_new = np.linspace(Ec['x'].iloc[0], Ec['x'].iloc[-1], 10001)
    
    x_new_Ec=pd.DataFrame(x_new)
    x_new_Ec[1]=f_Ec(x_new_Ec[0])-Ez
    x_new_Ec=x_new_Ec.loc[x_new_Ec[1]>=0]

    



    a1=x_new_Ec[0].iloc[0]
    b1=x_new_Ec[0].iloc[-1] 
        

#            
    I1=quad(lambda x :np.sqrt((2*m_n/hbar**2.0)*(f_Ec(x)-Ez)), a1, b1)[0]
        

 
    return np.exp(-2*I1)


def T_p(Ez):
    
    x_new = np.linspace(Ev['x'].iloc[0], Ev['x'].iloc[-1], 10001)
    
    x_new_Ev=pd.DataFrame(x_new)
    x_new_Ev[1]=f_Ev(x_new_Ev[0])-Ez
    x_new_Ev=x_new_Ev.loc[x_new_Ev[1]>=0]
#            x_new_Ev=x_new_Ev.loc[x_new_Ev[1]>=(b-a)/2]
    


#
    a2=x_new_Ev[0].iloc[0]
    b2=x_new_Ev[0].iloc[-1] 
        

#            
    I2=quad(lambda x :np.sqrt((2*m_p/hbar**2.0)*(f_Ev(x)-Ez)), a2, b2)[0]
        

 
    return np.exp(-2*I2)



keyword_array=['33']
J_dat=pd.DataFrame()
for keyword in keyword_array:
    #make a temp array of list of files within directory
    temp=[]
    
    
    #search for files containing specific keywords ie. 'cb.res' and 'vg_'
    for fname in os.listdir(directory):
        if keyword in fname:
            if 'cb.res' in fname:
                if 'vg_-' in fname:
        
                       temp.append(fname)
##
#    filelist=np.array(temp)
#
#    J=np.zeros(filelist.size)
#    for i, name in enumerate(filelist):
#        filename=name
#        #filename='40_AlGaN_UID_2_result.out.vg_-0.000-cb.res'
#        #filename='13nm_0.14dopedAlGaN_result.out.vg_-0.00-cb.res'
#        #filename='40nm_uidGaN_result.out.vg_-0.00-cb.res'
#        #filename='AlGaN_0.6_30nm_result.out.vg_-0.00-cb.res'
#        #filename='A3_10nmAlGN_result.out.vg_0.000-cb.res'
#        
#        dat=pd.read_csv(filename, delimiter=r"\s+", engine='python',header=None)
#        #dat=dat[(dat['x']>3.0e-6) & (dat['x']<(3.0e-6+thickness))].reset_index(drop=True)
#        labels=['x', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
#                     'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
#                     'uEv','uEv2','effective trap','stimulate R']
#        dat.columns = labels
#        #dat['X']=dat['x'].apply(lambda x: x*10e6)
#        
#        #    dat=dat[(dat['x']>5.0e-6) & (dat['x']<(5.0e-6+(thickness+1)*1e-7))].reset_index(drop=True)
#        dat['X']=dat['x'].apply(lambda x: x*10e6)
#        dat['x']=dat['x'].apply(lambda x: x*10e-2)
#    #    dat['bandgap']=dat['Ec']-dat['Ev']
#    #    dat['bandgap'].mean()
#    #    charge=dat['Electric field'].diff()/dat['X'].diff()
#    #    diffcharge=charge.diff()/dat['X'].diff()
#        #dat=dat.rename(columns={'Efp':'Hole Fermi Level'})
#        dat=dat.rename(columns={'Ec':'Conduction Band'})
#        dat=dat.rename(columns={'Ev':'Valence Band'})
#    #    #calculate depletion width
#    
#        a,b, a_index,b_index=find_depletion_width(dat)
#        
#        #create dataframe of Ec potential
#        Ec=dat[['x','Conduction Band']].copy()        
#        Ec=Ec.iloc[a_index:b_index].reset_index(drop=True)        
#
#        Ec['x']=Ec['x']-Ec['x'].iloc[0]
#        
#        Ec['x']=Ec['x'].values[::-1]
#        Ec=Ec.sort_values(['x','Conduction Band'],ascending=[True,True]).reset_index(drop=True)
##        Ec.plot('x','Conduction Band')
#
#
##        Ec=Ec.iloc[0:int((len(Ec)+1)/2)].reset_index(drop=True)
#        
#
#
#        
#
#        
#        
#        #create dataframe of Ev potential
#        Ev=dat[['x','Valence Band']].copy()        
#        Ev=Ev.iloc[a_index:b_index].reset_index(drop=True)
#
##       Ev['x']=Ev['x'].values[::-1]
#        
#
#        Ev=Ev.sort_values(['x','Valence Band'],ascending=[True,True]).reset_index(drop=True)
#
#        
#        Ev['x']=Ev['x']-Ev['x'].iloc[0]
#        
#        Ev['Valence Band']=-Ev['Valence Band']
#        
#        Ev['Valence Band']=Ev['Valence Band']-Ev['Valence Band'].min()
##        Ev.plot('x','Valence Band')
##        Ev=Ev.iloc[0:int((len(Ev)+1)/2)].reset_index(drop=True)
#        
#        
#        
#        #interpolate values in potential
#        f_Ec = interp1d(Ec['x'].values, Ec['Conduction Band'].values, kind='cubic')
#        f_Ev = interp1d(Ev['x'].values, Ev['Valence Band'].values, kind='cubic')
#        
#
#        
#        Efp=dat['Efp'].iloc[-2]
#        Efn=dat['Efn'].iloc[1]
#        
#        m=9.11e-31
#        kb=1.38e-23
#        Temp=300
#        h=1.09e-34
#        
#        #tunneling current prefactor
#        A_n=e*m_n*k*Temp/(2*np.pi**2*hbar**3)
#        A_p=e*m_p*k*Temp/(2*np.pi**2*hbar**3)
#        
#        I=lambda Ez: T(Ez)*np.log((1+np.exp(Efp-Ez)/k*300)/(1+np.exp(Efn-Ez)/k*300))
#        
#        max_Ec_energy=Ec['Conduction Band'].iloc[-1]
#        
#        J[i]=quad(lambda Ez: A_n*T_n(Ez)*np.log((1+np.exp(Efp-Ez)/k*300)/(1+np.exp(Efn-Ez)/k*300)), 0, Ec['Conduction Band'].iloc[-1])[0]\
#             -quad(lambda Ez: A_p*T_p(Ez)*np.log((1+np.exp(Efp-Ez)/k*300)/(1+np.exp(Efn-Ez)/k*300)), 0, Ev['Valence Band'].iloc[-1])[0]
#        
#        
#        
#    J_dat[keyword]=J
    
    
#
##plt.axvline(a)
##plt.axvline(b)
#
#host = host_subplot(111, axes_class=AA.Axes)
#plt.subplots_adjust(right=0.75)
#
#par1 = host.twinx()
#par2 = host.twinx()
#par3 = host.twinx()
#
#offset = 60
#new_fixed_axis = par2.get_grid_helper().new_fixed_axis
#par2.axis["right"] = new_fixed_axis(loc="right", axes=par2,
#                                        offset=(offset, 0))
#
#par2.axis["right"].toggle(all=True)
#
#offset = -50
#new_fixed_axis = par2.get_grid_helper().new_fixed_axis
#par3.axis["left"] = new_fixed_axis(loc="left", axes=par3,
#                                        offset=(offset, 0))
#
#par3.axis["left"].toggle(all=True)
##host.set_xlim(x_new[0],x_new[-1])
##host.set_ylim(charge[0],charge[-1])
#
#
#host.set_xlabel("Distance")
#host.set_ylabel("Charge")
#par1.set_ylabel("Charge Differential")
#par2.set_ylabel("Conduction Band")
#par3.set_ylabel("Electric field")
#
#p1, = host.plot(x_new*1e7, charge, label="Charge")
#p2, = par1.plot(x_new*1e7, diff_charge, label="Charge Differential")
#p3, = par2.plot(dat['x']*1e7, dat['Ec'], label="Conduction Band")
#p4, = par3.plot(x_new*1e7,electric_field_interpolated(x_new),label="Electric Field")
#
#
#host.legend()
#
#host.axis["left"].label.set_color(p1.get_color())
#par1.axis["right"].label.set_color(p2.get_color())
#par2.axis["right"].label.set_color(p3.get_color())
#par3.axis["left"].label.set_color(p4.get_color())
#
#plt.axvline(a*1e7)
#plt.axvline(b*1e7)
#plt.draw()
#plt.show()

#par1.set_ylim(0, 4)
#par2.set_ylim(1, 65)


##
#poly = np.polyfit(x_new,log_smooth_dc,5)
#poly_y = np.poly1d(poly)(x_new)
#
#plt.plot(x_new,poly_y)



#x_new=np.linspace(dat['x'].iloc[0],dat['x'].iloc[-1],1000000)
#fE = interp1d(dat['x'].values, dat['Electric field'].values, kind='cubic')
#
#window_size, poly_order = 10001, 4
#smooth_electric_field = savgol_filter(fE(x_new), window_size, poly_order)
#smooth_electric_field_interpolated = interp1d(x_new, smooth_electric_field, kind='cubic')
#
##plt.plot(x_new,smooth_electric_field_interpolated(x_new))
#
#charge=np.gradient(smooth_electric_field_interpolated(x_new))/np.gradient(x_new)
#
#window_size, poly_order = 17501, 4
#smooth_charge = savgol_filter(charge, window_size, poly_order)
#smooth_charge_interpolated = interp1d(x_new, smooth_charge, kind='cubic')
#plt.plot(x_new,smooth_charge_interpolated(x_new))


#plt.plot(x_new,charge)
#diffcharge=np.gradient(charge)/np.gradient(x_new)
#plt.plot(x_new,diffcharge)
#
#window_size, poly_order = 100001, 3
#smooth_dc = savgol_filter(diffcharge, window_size, poly_order)
#plt.plot(x_new,smooth_dc)
#(peak_index,info)=find_peaks(-smooth_dc,prominence=100000000)
#a=x_new[peak_index[0]]
#b=x_new[peak_index[-1]]
#plt.plot(x_new,fE(x_new))
#plt.scatter(x_new[peak_index],np.zeros(np.size(peak_index)))
#

#plt.plot(x_new)
#charge=dat['Electric field'].diff()/dat['x'].diff()
#diffcharge=charge.diff()/dat['x'].diff()
#dat=dat.rename(columns={'Efp':'Hole Fermi Level'})
#dat=dat.rename(columns={'Ec':'Conduction Band'})
#dat=dat.rename(columns={'Ev':'Valence Band'})
##calculate depletion width
#
#
##find turning point in band diagram by analysing peaks on the derivative of charge
#(peak_index,info)=find_peaks(-diffcharge,prominence=10000000)
#depletion_width=dat['x'].iloc[peak_index[1]]-dat['x'].iloc[peak_index[0]]
#a=dat['x'].iloc[peak_index[0]]
#b=dat['x'].iloc[peak_index[1]]
#
#
##create dataframe of Ec potential
#Ec=dat[['x','Conduction Band']].copy()
#
#Ec['x']=Ec['x'].values[::-1]
#Ec=Ec.sort_values(['x','Conduction Band'],ascending=[True,True]).reset_index(drop=True)
##Ec=Ec.iloc[peak_index[0]:peak_index[1]+1].reset_index(drop=False)
#
#Ec['x']=Ec['x']-Ec['x'].iloc[0]
#
##interpolate values in potential
#f = interp1d(Ec['x'].values, Ec['Conduction Band'].values, kind='cubic')
#x=Ec['x'].values
#x_new=np.linspace(Ec['x'].iloc[0],Ec['x'].iloc[-1],1000000)
#
#plt.plot(x,Ec['Conduction Band'].values)

filename=temp[24]
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

a,b, a_index,b_index=find_depletion_width(dat)

#create dataframe of Ec potential
Ec=dat[['x','Conduction Band']].copy()        
Ec=Ec.iloc[a_index:b_index].reset_index(drop=True)

#create dataframe of Ev potential
Ev=dat[['x','Valence Band']].copy()        
Ev=Ev.iloc[a_index:b_index].reset_index(drop=True)


Ec['x']=Ec['x'].values[::-1]
Ec=Ec.sort_values(['x','Conduction Band'],ascending=[True,True]).reset_index(drop=True)
Ec_section=Ec.loc[Ec['Conduction Band']<Ev['Valence Band'].max()]


Ec_section['x']=Ec_section['x']-Ec_section['x'].iloc[0]
Ec_section['Conduction Band']=Ec_section['Conduction Band']-Ec_section['Conduction Band'].min()

#Ec=Ec.iloc[0:int((len(Ec)+1)/2)].reset_index(drop=True)





#Ev['x']=Ev['x'].values[::-1]


Ev=Ev.sort_values(['x','Valence Band'],ascending=[True,True]).reset_index(drop=True)



Ev_section=Ev.loc[Ev['Valence Band']>Ec['Conduction Band'].min()]

Ev['Valence Band']=-Ev['Valence Band']
Ev['x']=Ev['x']-Ev['x'].iloc[0]
Ev['Valence Band']=Ev['Valence Band']-Ev['Valence Band'].min()





#Ev_section['Valence Band']=Ev_section['Valence Band']-Ev_section['Valence Band'].min()

Ev_section['Valence Band']=-Ev_section['Valence Band']
Ev_section['x']=Ev_section['x']-Ev_section['x'].iloc[0]
Ev_section['Valence Band']=Ev_section['Valence Band']-Ev_section['Valence Band'].min()

#Ev=Ev.iloc[0:int((len(Ec)+1)/2)].reset_index(drop=True)



#interpolate values in potential
f_Ec = interp1d(Ec['x'].values, Ec['Conduction Band'].values, kind='cubic')
f_Ev = interp1d(Ev['x'].values, Ev['Valence Band'].values, kind='cubic')

#
#def T_n(Ez):
#    
#    x_new = np.linspace(Ec['x'].iloc[0], Ec['x'].iloc[-1], 10001)
#    
#    x_new_Ec=pd.DataFrame(x_new)
#    x_new_Ec[1]=f_Ec(x_new_Ec[0])-Ez
#    x_new_Ec=x_new_Ec.loc[x_new_Ec[1]>=0]
#
#    
#
#
#
#    a1=x_new_Ec[0].iloc[0]
#    b1=x_new_Ec[0].iloc[-1] 
#        
#
##            
#    I1=quad(lambda x :np.sqrt((2*m_n/hbar**2.0)*(f_Ec(x)-Ez)), a1, b1)[0]
#        
#
# 
#    return np.exp(-2*I1)
#
#
#def T_p(Ez):
#    
#    x_new = np.linspace(Ev['x'].iloc[0], Ev['x'].iloc[-1], 10001)
#    
#    x_new_Ev=pd.DataFrame(x_new)
#    x_new_Ev[1]=f_Ev(x_new_Ev[0])-Ez
#    x_new_Ev=x_new_Ev.loc[x_new_Ev[1]>=0]
##            x_new_Ev=x_new_Ev.loc[x_new_Ev[1]>=(b-a)/2]
#    
#
#
##
#    a2=x_new_Ev[0].iloc[0]
#    b2=x_new_Ev[0].iloc[-1] 
#        
#
##            
#    I2=quad(lambda x :np.sqrt((2*m_p/hbar**2.0)*(f_Ev(x)-Ez)), a2, b2)[0]
#        
#
# 
#    return np.exp(-2*I2)

Efp=dat['Efp'].iloc[-2]
Efn=dat['Efn'].iloc[1]


#tunneling current prefactor
A_n=e*m_n*k*Temp/(2*np.pi**2*hbar**3)
A_p=e*m_p*k*Temp/(2*np.pi**2*hbar**3)

I=lambda Ez: T(Ez)*np.log((1+np.exp(Efp-Ez)/k*300)/(1+np.exp(Efn-Ez)/k*300))

#max_Ec_energy=Ec['Conduction Band'].iloc[-1]

J=quad(lambda Ez: A_n*T_n(Ez)*np.log((1+np.exp(Efp-Ez)/k*300)/(1+np.exp(Efn-Ez)/k*300)), Ec_section['Conduction Band'].min(), Ec_section['Conduction Band'].max())[0]\
     +quad(lambda Ez: A_p*T_p(Ez)*np.log((1+np.exp(Efp-Ez)/k*300)/(1+np.exp(Efn-Ez)/k*300)), Ev_section['Valence Band'].min(), Ev_section['Valence Band'].max())[0]
#        
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