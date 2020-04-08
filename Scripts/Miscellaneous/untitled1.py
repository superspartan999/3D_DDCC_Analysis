# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:57:13 2019

@author: Clayton
"""


import csv
import pandas as pd
import os
directory="D:\\1D-DDCC\\"
os.chdir(directory)
name="50nm_AlGaN_spacer_bottom_"
file=name+"result_ivn.csv"

data=pd.read_csv(file, delimiter=',')
data=data.reset_index(drop=False)
#with open(file) as f:
#    linelist=csv.reader(f,delimiter=",",index=False)