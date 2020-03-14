# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:55:35 2019

@author: Clayton
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from math import floor, sqrt




def SiLENSE(FileName):
    
    file=pd.read_csv(FileName,delimiter='	',header=None, engine='python')
    headers=file.iloc[0]
    
    newfile=pd.DataFrame(file.values[1:],columns=headers)
    
    plt.plot(newfile["Bias"],newfile["Current_density"])

    return newfile
    .
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

directory="C:\\Users\\Clayton\\Google Drive\\Research\\Simulations\\SiLENSe IV\\"
os.chdir(directory)
#tjdata=pd.read_csv('C:\\Users\\Clayton\\Google Drive\\Research\\Transport Structure Project\\Tunnel Junction IV\\SiO2\\052719AB\\200umr.csv')
#
#x=DDCC("IV curves 10nm AlGaN - Copy.ivn", "10nm")  
#y=DDCC("IV curves 30nm AlGaN - Copy.ivn", "30nm")
#z=DDCC("IV curves 50nm AlGaN - Copy.ivn", "50nm") 
#plt.plot(tjdata['V'],tjdata['I']/(np.pi*(200.0*(1e-4)/2)**2),label='Tunnel Junction')
#

#directory="D:\\1"
#
#os.chdir(directory)
#file=pd.read_csv(filename,header=None)
#x=OneD("10nmIV.csv",'1D Simulation')
##y=OneD("30nmIV.csv",'1D Simulation')
#z=OneD("50nmIV.csv",'1D Simulation')
#
#plt.grid(color='black')
#plt.xlim(-2,2)
#plt.ylim(-30,30)
#plt.legend()
#plt.tight_layout()
#plt.savefig('IV comparison for TJ.png')
#y=SiLENSE("IV_p_struc_10.dat")


#y=pd.read_csv("IV_p_struc_10.dat", delimiter='	')
#

#file=pd.read_csv("IV_p-doped-AlGaN.dat",delimiter='	',header=None, engine='python')
#headers=file.iloc[0]
#
#newfile=pd.DataFrame(file.values[1:],columns=headers)
#newfile=newfile.astype(float)
#negvalues=newfile.loc[newfile['Bias'] <0]['Current_density'].index
#newfile.loc[negvalues,'Current_density']=0
##
##
##
#plt.plot(newfile["Bias"],newfile["Current_density"], label='10nm SiLENSe')
##plt.plot(y["Bias"].values,y["Current_density"].values)
#file=pd.read_csv("IV_undoped-AlGaN_50nm.dat",delimiter='	',header=None, engine='python')
#headers=file.iloc[0]
#
#newfile=pd.DataFrame(file.values[1:],columns=headers)
#newfile=newfile.astype(float)
#negvalues=newfile.loc[newfile['Bias'] <0]['Current_density'].index
#newfile.loc[negvalues,'Current_density']=0
#
#plt.plot(newfile["Bias"],newfile["Current_density"], label='undoped')
##
#
###
#file=pd.read_csv("IV_p-doped-AlGaN_50nm.dat",delimiter='	',header=None, engine='python')
#headers=file.iloc[0]
#
#newfile=pd.DataFrame(file.values[1:],columns=headers)
#newfile=newfile.astype(float)
#negvalues=newfile.loc[newfile['Bias'] <0]['Current_density'].index
#newfile.loc[negvalues,'Current_density']=0
#
#
#plt.plot(newfile["Bias"],newfile["Current_density"], label='doped')
#
#plt.grid(color='black')
#plt.xlim(-2,2)
#plt.ylim(-4000,150000)
#plt.legend()
#plt.tight_layout()
#
##plt.savefig('IV comparison for SiLENSe (undoped).png')
#
#
#directory="D:\\1D-DDCC\\"
#os.chdir(directory)
#
##fig=plt.figure()
#name="Project_1_10_"
#file=name+"result_ivn.csv"
#ten=pd.read_csv(file, delimiter=',')
#ten=ten.reset_index(drop=False)
#
#plt.plot(ten['level_0'],ten['level_3'],label='10nm DDCC')
#
#name="Project_1_30_"
#file=name+"result_ivn.csv"
#
#thirty=pd.read_csv(file, delimiter=',',header=None)
#thirty=thirty.reset_index(drop=False)
#
##plt.plot(thirty[0],thirty[3],label='30nm DDCC')
##plt.plot(thirty['level_0'],thirty['level_3'],label='30nm')
#
#name="Project_1_50_"
#file=name+"result_ivn.csv"
#
#fifty=pd.read_csv(file, delimiter=',')
#fifty=fifty.reset_index(drop=False)

#plt.plot(fifty['level_0'],fifty['level_3'],label='50nm DDCC' )

directory="C:\\Users\\Clayton\\Google Drive\\Research\\Simulations\\3D IV Sim\\"
os.chdir(directory)
x=DDCC("IV curves 10nm AlGaN - copy.ivn", "10nm")  
y=DDCC("IV curves 30nm AlGaN - copy.ivn", "30nm")
z=DDCC("IV curves 50nm AlGaN - copy.ivn", "50nm") 

#
directory="D:\\1"
os.chdir(directory)
###file=pd.read_csv(filename,header=None)
##x=OneD("10nmIV.csv",'10nm thickness')
##y=OneD("30nmIV.csv",'30nm thickness')
#z=OneD("50nmIV.csv",'1D simulations')


plt.grid(color='black')
#plt.xlim(-2,2)
#plt.ylim(-4000,20000)
plt.legend()
#plt.tight_layout()

#z=silvaco("IV_silvaco_10nm.txt")
#
#plt.scatter(file[1],file[14]/(30e-7)**2)