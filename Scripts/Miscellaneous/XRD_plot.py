# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 14:07:06 2020

@author: Clayton
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy as scp
from scipy.signal import lfilter



directory="C:\\Users\\Clayton\\Downloads\\xrdfiles"
directory="C:\\Users\\Clayton\\Google Drive\\Research\\Transport Structure Project\\XRD Data"
directory="D:\\"
filelist=[]
for fname in os.listdir(directory):
            if '071720AA.csv' in fname:
               filelist.append(fname)
               
filename=filelist[0]


os.chdir(directory)
dat=pd.read_csv(filename, delimiter=",", engine='python',header=None, skiprows=27)  

new_header = dat.iloc[0] #grab the first row for the header
dat = dat.loc[1:] #take the data less the header row
dat.columns = new_header #set the header row as the df header

dat['Angle']=dat['Angle'].astype(float)
dat['Intensity']=dat['Intensity'].astype(float)

#plt.semilogy(dat['Angle'],dat['Intensity'])
plt.xlabel('Angle (°)')
plt.ylabel('Intensity (counts/sec)')

x=dat['Angle'].values
y=dat['Intensity'].values


#n = 3  # the larger n is, the smoother curve will be
#b = [1.0 / n] * n
#a = 1
#y = lfilter(b,a,y)

plt.semilogy(x,y)