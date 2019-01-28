# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:55:35 2019

@author: Clayton
"""
import pandas as pd
FileName= "IVmerged.txt"
file=pd.read_csv(FileName,delimiter='  ',header=None, engine='python')

