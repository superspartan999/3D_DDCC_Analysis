# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:59:20 2019

@author: Clayton
"""
from functions import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp


directory = 'D:\\1'
directory ='C:\\Users\\me_hi\\Downloads\\NTU-ITRI-DDCC-1D-3.4.7\\DDCC-1D'
os.chdir(directory)

def data(filename):
    
    headerlist= ['Position', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
             'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
             'uEv','uEv2','effective trap','layernumber', 'g']
    file=pd.read_csv(filename, sep="   ",header= None,engine='python')
    file.columns=headerlist
    return file

# def plotter(file,x_label='x_placeholder',y_label='y_placeholder',)

#Band Diagram

i=-0.0
i=round(i,2)
print(i)
file='Simple_LED.inp'
filename = str(file)+'_result.out.vg_'+str(i)+'00-cb.res'
os.chdir(directory)

fig=plt.figure()    
file=data(filename)
plt.plot(file['Position']/1e-7,file['Ec'], label='Ec doped AlGaN',color='red')
#plt.plot(file['Position']/1e-7,file['Efp'], label='Hole Fermi Level',color='blue')
plt.plot(file['Position']/1e-7,file['Ev'], label='Ev doped AlGaN',color='orange')
plt.xlabel('z (nm)')
plt.ylabel('Energy (eV)')
plt.tight_layout()
plt.xlim(30,60)
#plt.ylim(-7,10)
plt.legend(framealpha=100)
plt.grid()
#plt.savefig('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\'+str(filename)+'Valence.png')

i=round(i,2)
print(i)
file='Simple_LED.inp'
filename = str(file)+'_result.out.vg_'+str(i)+'00-cb.res'
os.chdir(directory)
headerlist= ['Position', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
              'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
              'uEv','uEv2','effective trap','layernumber', 'g']
file=pd.read_csv(filename, sep="   ",header= None,engine='python')
file.columns=headerlist

plt.plot(file['Position']/1e-7,file['Ec'], label='Ec undoped AlGaN', color='blue')
plt.plot(file['Position']/1e-7,file['Ev'], label='Ev undoped AlGaN', color='green')
plt.xlabel('z (nm)')
plt.ylabel('Energy (eV)')
plt.tight_layout()
#plt.xlim(0,1.6e-5)
#plt.ylim(-7,10)
plt.legend()
plt.grid()
#plt.savefig('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\'+str(filename)+'.png')