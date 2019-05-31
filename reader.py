# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:27:40 2019

@author: Clayton
"""
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
from functions import *


directory = 'E:\\10nmAlGaN\\Bias -42'
file= 'p_structure_0.17_10nm-out.vg_0.00.vd_-4.20.vs_0.00.unified'


directory= 'C:\\Users\\Clayton\\Google Drive\\Research\\Guillaume'
#
##directory= 'C:\\Users\\Clayton\\Desktop\\50nmAlGaN\\Bias -42'
#
#directory = "/Users/claytonqwah/Documents/Google Drive/Research/Transport Structure Project/3D data/10nmAlGaN/Bias -42"
file= 'LED4In-out.vg_0.00.vd_3.20.vs_0.00.unified'


os.chdir(directory)
df=pd.read_csv(file, delimiter=',')


sorted_data, xvalues,yvalues,zvalues= processdf(df)

    
G=graphfromdf(df,xvalues,yvalues,zvalues)   


#start=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == 0)]
#
#end=sorted_data.loc[(sorted_data['x'] == xvalues.iloc[int(len(xvalues)/2)][0])&(sorted_data['y'] == yvalues.iloc[int(len(yvalues)/2)][0])&(sorted_data['z'] == zvalues.iloc[len(zvalues)-1][0])]
##s=nx.shortest_path_length(G,start.index.values[0],end.index.values[0],weight='weight')
##h=nx.shortest_path(G,start.index.values[0],end.index.values[0],weight='weight')
#
##
#averagenodeenergy=averageenergy(G)
#
#
#path=pathfromnodes(h)
#    
#fig = plt.figure()
#
#ax = fig.add_subplot(111, projection='3d')
#
#x=path['x'].values
#
#y=path['y'].values
#
#z=path['z'].values
#
#ax.set_xlim(0, xvalues[0].iat[-1]) 
#ax.set_ylim(0,yvalues[0].iat[-1])
#ax.set_zlim(0,zvalues[0].iat[-1])
#
#ax.scatter(x, y, z, c='r', marker='o')
#
#
#save_json('C:\\Users\\Clayton\\Desktop\\Guillaume\\graph.js',G)
#
#
#
#k=read_json_file('C:\\Users\\Clayton\\Desktop\\Guillaume\\graph.json')



