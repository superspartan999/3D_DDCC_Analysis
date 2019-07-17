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


directory = 'C:\\Users\\Clayton\\Google Drive\\Research\\Transport Structure Project\\1D Band Structure\\1'
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
    
i=-0.0
i=round(i,2)
print(i)
filename = 'Heterostructure_10nm_result.out.vg_'+str(i)+'00-cb.res'
os.chdir(directory)
headerlist= ['Position', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
             'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
             'uEv','uEv2','effective trap','layernumber']
file=pd.read_csv(filename, sep="   ",header= None)
file.columns=headerlist
fig=plt.figure()
plt.plot(file['Position'],file['Ec'], label='Ec')
plt.plot(file['Position'],file['Ev'], label='Ev')
plt.xlabel('z (cm)')
plt.ylabel('Energy (eV)')
plt.xlim(0,2e-5)
plt.legend()