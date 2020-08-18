# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:11:03 2020

@author: Clayton
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,1,100)
a=(-0.077*x+3.189)
c13=5*x+103
c33=-32*x+405
e31=-0.11*x-0.49
e33=0.73*x+0.73
psp=-0.052*x-0.029
a0=3.189
psp0=-0.029
ppe=2*((a0-a)/a)*(e31-e33*(c13/c33))

sigma=abs(2*((a0-a)/a)*(e31-e33*(c13/c33))+psp-psp0)

plt.plot(x,sigma/1e4)
plt.plot(x, abs(psp)/1e4)
