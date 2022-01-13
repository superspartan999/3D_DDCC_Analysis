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
import scipy


directory='C:\\Users\\me_hi\\Downloads\\NTU-ITRI-DDCC-1D-3.4.7\\DDCC-1D'
filelist=[]
for fname in os.listdir(directory):
                if '-cb.res' in fname:
                    if 'GaNAlGaN.inp' in fname:
                
                       temp=fname
                       # temp=temp.replace('umr.csv','')
                       filelist.append(temp)
os.chdir(directory)
for filename in filelist:
# # filename='GaNAlGaN.inp_result.out.vg_'+str(i)+'.000-cb.res'
    dat=pd.read_csv(filename, delimiter=r"\s+", engine='python')
    
    dat.columns = ['x', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
              'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
              'uEv','uEv2','effective trap','stimulate R', 'layer']
    
    dat['X']=dat['x'].apply(lambda x: x*10e7)
    
    
    # plt.plot(dat["X"], dat['Ec'])
    plt.plot(dat["X"], dat['n'])
    # plt.plot(dat["X"], dat['Ec'], label=filename)
    # plt.plot(dat["X"], dat['Efn'])
    
    plt.xlabel('z(nm)', fontsize=12)
    
    plt.ylabel('V (eV)', fontsize=12)
    
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # plt.suptitle(filename, fontsize=20)
    
    # plt.savefig(filename + '.pdf')