# -*- coding: utf-8 -*-
"""
Created on Fri May 31 01:33:08 2019

@author: Kun
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

directory = 'E:\\10nmAlGaN\\Bias -42'
directory = 'E:\Google Drive\Research\Guillaume'
os.chdir(directory)


df=pd.read_csv("LED4In-out.vg_0.00.vd_3.20.vs_0.00.unified", delimiter=',')

sorted_data,xvalues,yvalues,zvalues=processdf(df)

zslice=extract_slice(sorted_data,'z',zvalues.iloc[348][0]/2,drop=True)

zmap=zslice[['x','y','Ec']]

fig = plt.figure()



x=zmap['x'].values

y=zmap['y'].values

z=zmap['Ec'].values

x_vals, x_idx = np.unique(x, return_inverse=True)
y_vals, y_idx = np.unique(y, return_inverse=True)

Ec_array = np.empty(x_vals.shape + y_vals.shape)

Ec_array.fill(np.nan)

Ec_array[x_idx, y_idx] = zmap['Ec'].values

plt.contour(x_vals,y_vals,Ec_array)

merged=np.vstack((x,y,z))

merged=np.transpose(merged)

dictm=dict(enumerate(merged,1))

G=nx.Graph()
G.add_nodes_from(enumerate(dictm,1))

G.add_nodes_from(dictm.keys())
for key, n in G.nodes.items()[:-1]:
    n['pos']=dictm[key][0:2].tolist()
    n['pot']=dictm[key][2]


