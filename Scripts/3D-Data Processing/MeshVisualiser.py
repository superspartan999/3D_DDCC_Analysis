# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:45:24 2020

@author: Clayton
"""
import os
import mayavi
import pandas as pd

directory = 'D:\\3D Simulations\\32nmAlGaN017\\Bias0'
file = 'p_structure_0.17_32nm-out.vg_0.00.vd_0.00.vs_0.00.unified'
os.chdir(directory)
df=pd.read_csv(file, delimiter=',')

node_map=df[['x','y','z']].copy()
#round up values in node map to prevent floating point errors
rounded_nodes=node_map.round(decimals=10)
#
##sort the nodes in ascending order
sorted_nodes=rounded_nodes.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
sorted_data=df.round({'x':10,'y':10,'z':10})
sorted_data=sorted_data.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)