# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:03:09 2020

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
import datetime


start = datetime.datetime.now()
m0=9.1e-31
m0_eV=(0.511*10**6)/(3.0e8)**2.0
hbar=6.6e-34
hbar_eV=4.1357e-15
#k=8.617e-5	
m_n=0.2*m0
m_n_eV=0.2*m0_eV
m_p=1.4*m0
#T=300
e=-1.6e-19



#function to find nearest array point to specific value
def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx],idx
    
    
##look for depletion width in structure. this is done by calculating the double differential of the electric field and finding the peak
#def find_depletion_width(dat):
#
#    #create new x-array for interpolation
#    x_new=np.linspace(dat['x'].iloc[0],dat['x'].iloc[-1],1000)
#    
#
#    #interpolate electric field
#    electric_field_interpolated = interp1d(dat['x'].values, dat['Electric field'].values, kind='cubic')
#
#    #calculate charge from electric field gradient
#    charge=np.gradient(electric_field_interpolated(x_new))/np.gradient(x_new)
#
#    #calculate charge gradient
#    diff_charge=np.gradient(charge)/np.gradient(x_new)
#
#    #find midpoint of junction
#    mid_x_ind=np.argsort(x_new)[len(x_new)//2]
#    
#    #find peaks in the charge gradient which correspond to the turning point in band diagram
#    (peak_index,info)=find_peaks(-diff_charge,prominence=1000, width=10)
#    
#    
#    #calculate distance of peak from center
#    peak_distance=x_new[peak_index]-x_new[mid_x_ind]
#    
#    #find peaks closest to the center
#    peak_distance=abs(x_new[peak_index]-x_new[mid_x_ind])
#    window= np.argpartition(peak_distance,1)
#    
#    a=x_new[peak_index[window[0]]]
#    b=x_new[peak_index[window[1]]]
#    
#    return a,b, peak_index[window[0]],peak_index[window[1]]


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
directory='D:/1D-DDCC'


def simpson(f, a, b, n):
    h=(b-a)/n
    k=0.0
    x=a + h
    for i in range(1,int(n/2 + 1)):
        k += 4*f(x)
        x += 2*h

    x = a + 2*h
    for i in range(1,int(n/2)):
        k += 2*f(x)
        x += 2*h
    return (h/3)*(f(a)+f(b)+k)

def T_n(Ez):   
    x_new = np.linspace(Ec['x'].iloc[0], Ec['x'].iloc[-1], 10001)
    #    
    g=lambda x:f(x)-Ez
    
    Allowed=pd.DataFrame({'x':x_new,'g':g(x_new)})
    Allowed=Allowed.loc[Allowed['g']>0]
    plt.plot(x_new,f(x_new))
    plt.plot(Allowed['x'].values,f(Allowed['x'].values))
    
    if (len(Allowed))>0:
#        integrand= lambda x :np.sqrt((2*m_n/hbar**2.0)*abs(e)*g(x))
        integrand= lambda x :np.sqrt((2*m_n_eV/hbar_eV**2.0)*g(x))

    
        a1=Allowed['x'].iloc[0]
        b1=Allowed['x'].iloc[-1]#                
        I1=simpson(integrand,a1,b1,1000)
        return np.exp(-2*I1)
        
    else: 
        return 0
#    print(len(Allowed),Ez)
    



def T_p(Ez):
    
    x_new = np.linspace(Ec['x'].iloc[0], Ec['x'].iloc[-1], 10001)
    
    g=lambda x:Ez-f_p(x)
    
    Allowed=pd.DataFrame({'x':x_new,'g':g(x_new)})
    Allowed=Allowed.loc[Allowed['g']>0]
    integrand= lambda x :np.sqrt((2*m_p/hbar**2.0)*abs(e)*g(x))
    

    a1=Allowed['x'].iloc[0]
    b1=Allowed['x'].iloc[-1]#                
    I1=simpson(integrand,a1,b1,100)
    
    return np.exp(-2*I1)

def T(Ez):   
    x_new = np.linspace(Ec['x'].iloc[0], Ec['x'].iloc[-1], 10001)
        
    x_new_pd=pd.DataFrame(x_new)
    x_new_pd[1]=f(x_new_pd[0])-Ez
    x_new_pos=x_new_pd.loc[x_new_pd[1]>0]
#        x_new_pos[0]=x_new_pos[0].round(20)
#        plt.plot(x_new_pos[0],integrand(x_new_pos[0]))
        
    
    if len(x_new_pos)>0:
    
            a1=x_new_pos[0].iloc[0]
            b1=x_new_pos[0].iloc[-1] 
            
            g=lambda x:f(x)-Ez
            integrand= lambda x :np.sqrt((2*m_n/hbar**2.0)*abs(e)*g(x))
            
            I1=simpson(integrand,a1,b1,5)
            
#            plt.plot(x_new_pos[0],integrand(x_new_pos[0]))
            

            return I1

os.chdir(directory)
keyword_array=['11','22','33']
#keyword_array=['11']
V=np.linspace(0.2,5,25)
J_dat=pd.DataFrame()

for keyword in keyword_array:
    #make a temp array of list of files within directory
    temp=[]

    
    #search for files containing specific keywords ie. 'cb.res' and 'vg_'
    for fname in os.listdir(directory):
        if keyword in fname:
            if 'cb.res' in fname:
                if 'vg_-' in fname:
#                   if '1.000' in fname:
#    
                        temp.append(fname) 



                    
    J_n=np.zeros(len(temp))
    J_p=np.zeros(len(temp))
    for i, name in enumerate(temp):
        filename=name
        
        dat=pd.read_csv(filename, delimiter=r"\s+", engine='python',header=None)
        labels=['x', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
                     'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
                     'uEv','uEv2','effective trap','stimulate R']
        dat.columns = labels
        dat['Electric field']=-dat['Electric field']
        dat['X']=dat['x'].apply(lambda x: x*10e6)
        dat['x']=dat['x'].apply(lambda x: x*10e-2)

        dat=dat.rename(columns={'Ec':'Conduction Band'})
        dat=dat.rename(columns={'Ev':'Valence Band'})
        #    #calculate depletion width
        
        a,b, a_index,b_index=extractDepletion(dat)
#        plt.plot()
#sghe
#        plt.axvline(a)
#        plt.axvline(b)
        
    #    a,b, a_index,b_index=find_depletion_width(dat)
        
        #create dataframe of Ec potential
        Ec=dat[['x','Conduction Band']].copy()
        Ev=dat[['x','Valence Band']].copy()
    
        Ec['x']=Ec['x'].values[::-1]
        Ec=Ec.sort_values(['x','Conduction Band'],ascending=[True,True]).reset_index(drop=True)
        Ec=Ec.iloc[a_index:b_index].reset_index(drop=False)
        Ec['x']=Ec['x']-Ec['x'].iloc[0]

#        plt.plot(Ec['x'],Ec['Conduction Band'])
#        Ec=Ec.loc[Ec['Conduction Band']<Ev['Valence Band'].max()]

        
        Ev['x']=Ev['x'].values[::-1]
        Ev=Ev.sort_values(['x','Valence Band'],ascending=[True,True]).reset_index(drop=True)
        Ev=Ev.iloc[a_index:b_index].reset_index(drop=False)
        Ev['x']=Ev['x']-Ev['x'].iloc[0]
        
#        Ec=Ec.loc[Ec['Conduction Band']<Ev['Valence Band'].max()]
        
#        Ec['Conduction Band']=Ec['Conduction Band'].loc[Ec['Conduction Band']<Ev['Valence Band'].max()]
        if len(Ec)>0:
            x_new=np.linspace(Ec['x'].iloc[0],Ec['x'].iloc[-1])
        
            #interpolate values in potential
            f = interp1d(Ec['x'].values, Ec['Conduction Band'].values,kind='cubic')
            
    #
    #        coeff=np.polyfit(Ec['x'].values, Ec['Conduction Band'].values ,1)
    #        f=np.poly1d(coeff)
    #        plt.plot(x_new,f(x_new))
            f_p =interp1d(Ev['x'].values, Ev['Valence Band'].values, kind='cubic')
    #        plt.plot(x_new,f_p(x_new))
            Efp=dat['Efp'].iloc[-2]
            Efn=dat['Efn'].iloc[1]
            
            #m=9.11e-31
            kb=1.38e-23
            Temp=300
            #h=1.09e-34
            
            #tunneling current prefactor
            A=e*m_n*kb*Temp/(2*np.pi**2*hbar**3)
            k=8.6173e-5
            m_ev=0.2*(0.511*10**6)/(3.0e8)**2.0
            h=4.1357e-15
#            A=e*m_ev*k*Temp/(2*np.pi**2*h**3)
            x_new = np.linspace(Ec['x'].iloc[0], Ec['x'].iloc[-1], 10001)
            print(i)
            
    #        J[i]=quad(lambda Ez: A*T_test(Ez)*np.log((1+np.exp(abs(e)*(Efp-Ez)/(kb*Temp)))/(1+np.exp(abs(e)*(Efn-Ez)/(kb*Temp)))) , Ec['Conduction Band'].min(), Ev['Valence Band'].max())[0]
            integrand_n=lambda Ez: A*T_n(Ez)*np.log((1+np.exp(abs(e)*(Efp-Ez)/(kb*Temp)))/(1+np.exp(abs(e)*(Efn-Ez)/(kb*Temp))))
#            integrand_n=lambda Ez: T_n(Ez)*np.log((1+np.exp(abs(e)*(Efp-Ez)/(kb*Temp)))/(1+np.exp(abs(e)*(Efn-Ez)/(kb*Temp))))
#            integrand_n=lambda Ez: T_n(Ez)
            interval_1=Ec['Conduction Band'].min()
            interval_2=Ev['Valence Band'].max()
            Ez=np.linspace( Ec['Conduction Band'].min(), Ev['Valence Band'].max(),1000)
            print(b-a)
    #        for E in Ez:
    #            plt.plot(E,T(E))
    #        print(interval_2-interval_1)
            if (interval_2-interval_1) > 0:
              J_n[i]=simpson(integrand_n , Ec['Conduction Band'].min(), Ev['Valence Band'].max(),100)
    #           J_n[i]=T_n(Ev['Valence Band'].max())
    #            plt.
    #            J_n[i]=b-a
            
            else:
                J_n[i]=0
            
        else:
            J_n[i]=0
            
#        J[i]=T_test(Ev['Valence Band'].max())
    J_dat[keyword]=J_n

finish = datetime.datetime.now()

print(finish-start)