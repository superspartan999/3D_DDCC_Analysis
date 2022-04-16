# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:33:15 2022

@author: Clayton
"""
import numpy as np
import pandas as pd
N=10207
nf=1020
p=0.5
Nb=10



# expect=nf*p**Nb

expected_list=np.array([])

class_num=11


prob_list=pd.DataFrame(0, index=np.arange(class_num),columns=['i','P','Exp'])
for i in np.arange(0,class_num-1):
    perm=np.math.factorial(int(Nb))/(np.math.factorial(int(i))*np.math.factorial(int(Nb-i)))
    print(i)
    P=perm*(p**i)*((1-p)**(Nb-i))
    prob_list['i'].loc[i]=i
    prob_list['P'].loc[i]=P
    expect=nf*P
    # expect=expect*(p/(1-p))*(Nb+1-i)/i
    expected_list=np.append(expected_list,expect)