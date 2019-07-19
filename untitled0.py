# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:39:38 2019

@author: Clayton
"""
from functions import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from math import floor, sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import os
import networkx as nx
from networkx.readwrite import json_graph
import simplejson as json
import heapq
import csv

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
directory = 'D:\\1D-DDCC'
directory ='C:\\Users\\Clayton\\Google Drive\Research\\Transport Structure Project\\1D Band Structure\\1'
directory ='C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\1D-DDCC'

#filedict={}
#for i in np.arange(-5,5,1.0,dtype=float):
#    i=round(i,2)
#    print(i)
#    filename = 'TJ_Full_result.out.vg_'+str(i)+'00-cb.res'
#    os.chdir(directory)
#    headerlist= ['Position', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
#                 'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
#                 'uEv','uEv2','effective trap','layernumber']
#    file=pd.read_csv(filename, sep="   ",header= None)
#    file.columns=headerlist
#    filedict.update({i:file})
#    plt.plot(file['Position'],file['Ec'])
#    plt.plot(file['Position'],file['Ev'])
array=[]
os.chdir(directory)
#with open("Project_1_30_doped_result.ivn.iv") as csvfile:
#    reader = csv.reader(csvfile) # change contents to floats
#    for row in reader: # each row is a list
#        array.append(row)
#
#df=pd.DataFrame(array)
#df2=df[0].str.split('  ',expand=True)
#df3=df2.iloc[1:].shift(-1,axis=1)
#df4=df3.iloc[52:].shift(-1,axis=1)
#df3=df3.iloc[0:50]
#df4=df4.dropna(axis=1)
#df3=df3.dropna(axis=1)
#
#def drop_empty(df4):
#    temp=df4[df4.isin([''])]
#    temp=temp.dropna(axis=1)
#    colstodrop=temp.columns.values
#    df4=df4.drop(colstodrop,axis=1)
#
#    df4.columns = range(df4.shape[1])
#    
#    
#    return df4
#dat_1=drop_empty(df3)
#
#dat_2=drop_empty(df4)
#
#final=pd.concat([dat_1,dat_2])
##df3.iloc[51:]=df4

#x=OneD("10nm_doped_spacer.csv",'10nm')
#y=OneD("30nm_doped_spacer.csv",'30nm')
#z=OneD("50nm_doped_spacer.csv",'50nm')
###
##
#plt.grid(color='black')
#plt.xlim(-0.5,0.5)
#plt.ylim(-5000,5000)
#plt.legend()
#plt.xlabel('Voltage (V)')
#plt.ylabel('Current Density (A/cm^2)')
#plt.tight_layout()
#plt.savefig('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Doped AlgaN with spacer.png')
##
#
#fig=plt.figure()
#x=OneD("10nm_undoped_spacer.csv",'Undoped')
#y=OneD("10nm_doped_spacer.csv",'Doped')
#
#plt.legend()
#plt.grid(color='black')
#plt.xlabel('Voltage (V)')
#plt.ylabel('Current Density (A/cm^2)')
##plt.xlim(-2,1)
#plt.ylim(-3000,3000)
#plt.tight_layout()
#plt.savefig\
#('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Doped vs Undoped AlgaN with spacer (10nm).png')
#
#fig=plt.figure()
#
#x=OneD("30nm_undoped_spacer.csv",'Undoped')
#y=OneD("30nm_doped_spacer.csv",'Doped')
#
#
#
#plt.legend()
#plt.grid(color='black')
#plt.xlabel('Voltage (V)')
#plt.ylabel('Current Density (A/cm^2)')
##plt.xlim(-4,0.5)
#plt.ylim(-5000,5000)
#plt.tight_layout()
#plt.savefig\
#('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Doped vs Undoped AlgaN with spacer (30nm).png')
#
#
#fig=plt.figure()
#x=OneD("50nm_undoped_spacer.csv",'Undoped')
#y=OneD("50nm_doped_spacer.csv",'Doped')
#
#plt.legend()
#plt.grid(color='black')
#plt.xlabel('Voltage (V)')
#plt.ylabel('Current Density (A/cm^2)')
##plt.xlim(-4,0.5)
#plt.ylim(-5000,5000)
#plt.tight_layout()
#plt.savefig\
#('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Doped vs Undoped AlgaN with spacer (50nm).png')
#
#
#fig=plt.figure()
#x=OneD("50nm_doped.csv",'No Spacer')
#y=OneD("50nm_doped_spacer.csv",'Spacer')
#
#plt.legend()
#plt.grid(color='black')
#plt.xlabel('Voltage (V)')
#plt.ylabel('Current Density (A/cm^2)')
#plt.xlim(-2,2)
#plt.ylim(-5000,5000)
#plt.tight_layout()
#plt.savefig\
#('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Spacer vs No Spacer Doped AlGaN (50nm).png')
#
fig=plt.figure()
x=OneD("10nm_doped.csv",'No Spacer')
y=OneD("10nm_doped_spacer.csv",'Spacer')

plt.legend()
plt.grid(color='black')
plt.xlabel('Voltage (V)')
plt.ylabel('Current Density (A/cm^2)')
plt.xlim(-2,2)
plt.ylim(-5000,5000)
plt.tight_layout()
plt.savefig\
('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Spacer vs No Spacer Doped AlGaN (10nm).png')

fig=plt.figure()
x=OneD("30nm_doped.csv",'No Spacer')
y=OneD("30nm_doped_spacer.csv",'Spacer')

plt.legend()
plt.grid(color='black')
plt.xlabel('Voltage (V)')
plt.ylabel('Current Density (A/cm^2)')
plt.xlim(-2,2)
plt.ylim(-5000,5000)
plt.tight_layout()
plt.savefig\
('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Spacer vs No Spacer Doped AlGaN (30nm).png')

fig=plt.figure()
x=OneD("50nm_doped.csv",'No Spacer')
y=OneD("50nm_doped_spacer.csv",'Spacer')

plt.legend()
plt.grid(color='black')
plt.xlabel('Voltage (V)')
plt.ylabel('Current Density (A/cm^2)')
plt.xlim(-2,2)
plt.ylim(-5000,5000)
plt.tight_layout()
plt.savefig\
('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Spacer vs No Spacer Doped AlGaN (50nm).png')
#file=pd.read_csv("50nm_undoped.csv",header=None)
#fig=plt.figure()
#a=OneD("10nm_undoped.csv",'10nm (No Spacer)')
#b=OneD("10nm_Undoped_spacer.csv",'10nm (With Spacer)')
#c=OneD("30nm_undoped.csv",'30nm (No Spacer)')
#d=OneD("30nm_Undoped_spacer.csv",'30nm (With Spacer)')
#e=OneD("50nm_undoped.csv",'50nm (No Spacer)')
#f=OneD("50nm_Undoped_spacer.csv",'50nm (With Spacer)')
#
#plt.legend()
#plt.grid(color='black')
#plt.xlabel('Voltage (V)')
#plt.ylabel('Current Density (A/cm^2)')
#plt.xlim(-8,6)
#plt.ylim(-5000,5000)
#plt.tight_layout()
#plt.savefig\
#('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Spacer vs No Spacer Undoped AlGaN (10nm).png')
##
#fig=plt.figure()
#x=OneD("30nm_undoped.csv",'No Spacer')
#y=OneD("30nm_Undoped_spacer.csv",'Spacer')
#
#plt.legend()
#plt.grid(color='black')
#plt.xlabel('Voltage (V)')
#plt.ylabel('Current Density (A/cm^2)')
#plt.xlim(-5,6)
#plt.ylim(-5000,5000)
#plt.tight_layout()
#plt.savefig\
#('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Spacer vs No Spacer Undoped AlGaN (30nm).png')
##Band Diagram
#fig=plt.figure()    
#i=-0.0
#i=round(i,2)
#print(i)
#file='TJ_Full_95_13'
#filename = str(file)+'_result.out.vg_'+str(i)+'00-cb.res'
#os.chdir(directory)
#headerlist= ['Position', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
#             'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
#             'uEv','uEv2','effective trap','layernumber']
#file=pd.read_csv(filename, sep="   ",header= None,engine='python')
#file.columns=headerlist
#
#plt.plot(file['Position']/1e-7,file['Ec'], label='Ec')
#plt.plot(file['Position']/1e-7,file['Ev'], label='Ev')
#plt.xlabel('z (nm)')
#plt.ylabel('Energy (eV)')
#plt.tight_layout()
##plt.xlim(0,1.6e-5)
##plt.ylim(-7,10)
#plt.legend()
#plt.grid()
#plt.savefig('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\'+str(filename)+'.png')