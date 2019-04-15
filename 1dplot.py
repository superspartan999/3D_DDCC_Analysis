# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:13:58 2019

@author: Clayton
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

directory="D:\\1D-DDCC"
os.chdir(directory)
colnames=['x', 'Ec', 'Ev', 'Efn','Efp','n','p','Jn','Jp','Rad','nonRad','Auger', 'RspPL', 'eb', 'ebh' 'gen', 'active dopants','impurity'
                                                                           '1/u_Ec', '1/u_Ev', '1/u_Ehh','Electic field', 'mu_n','mu_p','uEc','uEv',
                                                                           'uEv2','effective traps','layernum']
colnames={'0':'x', '1':'Ec', '2':'Ev', '3':'Efn','4':'Efp','5':'n','6':'p','7':'Jn','8':'Jp','9':'Rad','10':'nonRad','11':'Auger', '12':'RspPL', '13':'eb', '14':'ebh','15':'gen', '16':'active dopants','17':'impurity','18':'1/u_Ec', '19':'1/u_Ev', '20':'1/u_Ehh','21':'Electic field', '22':'mu_n','23':'mu_p','24':'uEc','25':'uEv','26':'uEv2','27':'effective traps','28':'layernum'}
df=pd.read_csv('TJ_11_result.out.vg_0.00-cb.res', delimiter='   ',header=None,engine='python')
df1=df.rename(columns=colnames  )


