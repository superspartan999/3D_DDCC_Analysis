# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:37:17 2019

@author: Clayton
"""
import os
import pandas as pd
import numpy as np
directory='D:/16nmAlGaN'
os.chdir(directory)

file='EBLAlavg.out'
comp_header=10
#comp=pd.read_csv(file, sep='    ',names=['z','Comp'])
comp_map_info=pd.read_csv('Al_map.out',nrows=comp_header)
num_nodes = int(comp_map_info.iloc[comp_header-1, 0])

my_data = pd.read_csv('Al_map.out', skiprows=head_len, nrows=num_nodes, 
                          delim_whitespace=True, header=None, names=['Node','Comp'], 
                          engine='python')