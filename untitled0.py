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
    
    file=pd.read_csv(FileName,delimiter=',',header=None, engine='python')
    headers=file.iloc[0]
    
    newfile=pd.DataFrame(file.values[1:],columns=headers)
    
    plt.plot(newfile["Bias"],newfile["Current_density"])

    return newfile
    
def OneD(FileName, legend_label):
#    file=pd.read_csv(FileName, delimiter=',',header=None)
    file=pd.read_csv(FileName, delimiter=',')
    file=file.reset_index(drop=False)

    
#    plt.plot(file['level_0'],file['level_3'],label=legend_label)
    
    
    
    return file
       
def DDCC(FileName, legend_label):
    
    file=pd.read_csv(FileName,delimiter='  ',header=None, engine='python')

    plt.plot(file['level_0'],file['level_3']/(30e-7)**2,label=legend_label)
    
    
    
    return file

def silvaco(FileName):
    
    file=pd.read_csv(FileName,delimiter='	',header=None, engine='python')
    
    plt.plot(-file[5],file[3])
    
    return file
directory = 'D:\\1'
#directory ='C:\\Users\\Clayton\\Google Drive\Research\\Transport Structure Project\\1D Band Structure\\1'
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

##file=pd.read_csv('Project_1_10_doped_result.ivn.iv', sep='  |   |    ', header=None)

directory="D:\\1D-DDCC\\"
os.chdir(directory)
#
#ten=OneD('10nm_doped.csv','10nm')
#thirty=OneD('30nm_doped.csv', '30nm')
#fifty=OneD('50nm_doped.csv', '50nm')
#
#plt.plot(ten[0],ten[3],label='10nm')
#
#plt.plot(thirty[0],thirty[3],label='30nm')
#
#
#plt.plot(fifty[0],fifty[3],label='50nm')

#x=OneD('10nm_undoped.csv','Undoped')
#y=OneD('10nm_doped.csv','Doped')
#
#plt.plot(x[0],x[3],label='Undoped')
#plt.plot(y[0],y[3],label='Doped')
#
#plt.grid(color='black')
#plt.xlim(-0.5,0.5)
#plt.ylim(-10000,10000)
#plt.legend()
#plt.xlabel('Voltage (V)')
#plt.ylabel('Current Density (A/cm^2)')
#plt.tight_layout()
###plt.savefig('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Doped AlGaN Comparison.png')

tjdata=pd.read_csv('C:\\Users\\Clayton\\Google Drive\\Research\\Transport Structure Project\\Tunnel Junction IV\\SiO2\\052719AB\\200umr.csv')
fig=plt.figure()
name="Project_1_10_"
file=name+"result_ivn.csv"
ten=pd.read_csv(file, delimiter=',')
ten=ten.reset_index(drop=False)

plt.plot(ten['level_0'],ten['level_3'],label='10nm')

name="Project_1_30_"
file=name+"result_ivn.csv"

thirty=pd.read_csv(file, delimiter=',',header=None)
thirty=thirty.reset_index(drop=False)

plt.plot(thirty[0],thirty[3],label='30nm')
#plt.plot(thirty['level_0'],thirty['level_3'],label='30nm')

name="Project_1_50_"
file=name+"result_ivn.csv"

fifty=pd.read_csv(file, delimiter=',')
fifty=fifty.reset_index(drop=False)

plt.plot(fifty['level_0'],fifty['level_3'],label='50nm')
plt.plot(tjdata['V'],tjdata['I']/(np.pi*(200.0*(1e-4)/2)**2),label='Tunnel Junction')
###file=name+"result_ivn.csv"
####
#####
#####z=OneD(file,'50nm')
####
#####
plt.grid(color='black')
plt.xlim(-2.5,2.5)
plt.ylim(-30,30)


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

#fig=plt.figure()
#x=OneD("10nm_doped.csv",'No Spacer')
#y=OneD("10nm_doped_spacer.csv",'Spacer')
#
#plt.legend()
#plt.grid(color='black')
#plt.xlabel('Voltage (V)')
#plt.ylabel('Current Density (A/cm^2)')
#plt.xlim(-2,2)
#plt.ylim(-5000,5000)
#plt.tight_layout()
#plt.savefig\
#('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Spacer vs No Spacer Doped AlGaN (10nm).png')
#
#fig=plt.figure()
#x=OneD("30nm_doped.csv",'No Spacer')
#y=OneD("30nm_doped_spacer.csv",'Spacer')
#
#plt.legend()
#plt.grid(color='black')
#plt.xlabel('Voltage (V)')
#plt.ylabel('Current Density (A/cm^2)')
#plt.xlim(-2,2)
#plt.ylim(-5000,5000)
#plt.tight_layout()
#plt.savefig\
#('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Spacer vs No Spacer Doped AlGaN (30nm).png')
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

plt.tight_layout()
plt.savefig('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\IV comparison for TJ.png')

plt.xlim(-2,2)
plt.ylim(-5000,5000)
plt.tight_layout()
plt.savefig\
('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Spacer vs No Spacer Doped AlGaN (30nm).png')


#plt.tight_layout()
#
###
##
fig=plt.figure()

x=OneD("50nm_doped.csv",'No Spacer')
y=OneD("50nm_doped_spacer.csv",'Spacer')


name="Project_1_50_doped_"
file=name+"result_ivn.csv"
#file='10nm.csv'
#x=OneD(file, 'Landscape')
x=pd.read_csv(file, delimiter=',')
x=x.reset_index(drop=False)
#name="Project_1_10_"
#file=name+"result_ivn.csv"
#y=OneD(file,'No Spacer')
#y=pd.read_csv(file, delimiter=',',header=None)
name="Project_1_50_doped_no_landscape_"
file=name+"result_ivn.csv"
z=pd.read_csv(file, delimiter=',')
z=z.reset_index(drop=False)

plt.plot(x['level_0'],x['level_3'],label='tunneling')
#plt.plot(y[0],y[3],label='not doped')
#plt.plot(y['level_0'],y['level_3'],label='not doped')
plt.plot(z['level_0'],z['level_3'],label='no tunneling')
#
plt.legend()
plt.grid(color='black')
plt.xlabel('Voltage (V)')
plt.ylabel('Current Density (A/cm^2)')

plt.xlim(-1,1)
#plt.ylim(-5000,5000)
plt.tight_layout()
plt.savefig('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Low Doped vs No Dope IV 30nm.png')
###

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

#
##directory="C:\\Users\\Clayton\\Google Drive\\Research\\Simulations\\1D-DDCC"
#plt.savefig\
#('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\Spacer vs No Spacer Undoped AlGaN (30nm).png')

file='Project_1_30_doped'
filename = str(file)+'_result.out.vg_'+str(i)+'00-cb.ivn.csv'
fig=plt.figure()
undoped=pd.read_csv("30nm_undoped.csv",header=None)

plt.plot(undoped[0],undoped[3],label='Undoped AlGaN')

doped=pd.read_csv("30nm_doped.csv",header=None)

plt.plot(doped[0],doped[3],label='Doped AlGaN')

plt.xlabel('Voltage (V)')
plt.ylabel('Current Density (A/cm^2)')
plt.tight_layout()
plt.xlim(-4,2)
plt.ylim(-5000,5000)
#plt.ylim(-7,10)
plt.legend(framealpha=100)
plt.grid()

##Band Diagram
#fig=plt.figure()    
#i=-0.0
#i=round(i,2)
#print(i)
#file='TJ_Full_95_13'
#filename = str(file)+'_result.out.vg_'+str(i)+'00-cb.res'

#os.chdir(directory)
fig=plt.figure()    
i=-0.0
i=round(i,2)
print(i)
file='Project_1_10_doped'

filename = str(file)+'_result.out.vg_'+str(i)+'00-cb.res'
os.chdir(directory)
headerlist= ['Position', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
             'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
             'uEv','uEv2','effective trap','layernumber']
file=pd.read_csv(filename, sep="   ",header= None,engine='python')
file.columns=headerlistHEAD

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
directory = 'D:\\1'
os.chdir(directory)
i=round(i,2)
print(i)
file='Project_1'
#
#plt.plot(file['Position']/1e-7,file['Ec'], label='Ec')
##plt.plot(file['Position']/1e-7,file['Ev'], label='Landscape')
plt.plot(file['Position']/1e-7,file['Efp'], label='Efp undoped',color='blue')
plt.plot(file['Position']/1e-7,file['Ev'], label='Ev undoped',color='green')
plt.xlabel('z (nm)')
plt.ylabel('Energy (eV)')
plt.tight_layout()
#plt.xlim(30,80)

#plt.ylim(-7,10)
plt.legend()

#plt.savefig('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\Valence AlGaN Bias0 No Dope 50nm')
##
#fig=plt.figure()
i=round(i,2)
print(i)
file='Project_1_10_doped'
filename = str(file)+'_result.out.vg_'+str(i)+'00-cb.res'
os.chdir(directory)
headerlist= ['Position', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
             'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
             'uEv','uEv2','effective trap','layernumber']
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
plt.plot(file['Position']/1e-7,file['Efp'], label='Efp doped', color='Red')#,color='blue')
#plt.plot(file['Position']/1e-7,file['Ec'], label='Ec',color='green')
plt.plot(file['Position']/1e-7,file['Ev'], label='Ev doped',color='orange')
plt.xlabel('z (nm)')
plt.ylabel('Energy (eV)')
plt.tight_layout()
plt.xlim(30,60)
#plt.ylim(-7,10)
plt.legend()
plt.grid()
plt.savefig('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\Dope vs No Dope Bias0 10nm')

