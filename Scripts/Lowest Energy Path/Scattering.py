# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:00:52 2020

@author: Clayton
"""
from functions import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp


comp=1
length=20
material='AlGaN'
#
directory='D:/'+str(length)+'nm'+material+''+str(comp)

directory='D:\\Guillaume Data\\LEDIndiumCompo_'+str(comp)+'Al_'+str(length)+'Angs_\\Bias3'
os.chdir(directory)

#file= directory+'/'+material+'_0.5_'+str(length)+'nm_-out.vg_0.00.vd_0.00.vs_0.00.unified'
file='LEDIndiumCompo_'+str(comp)+'Al_'+str(length)+'Angs_-out.vg_0.00.vd_0.00.vs_0.00.unified'

os.chdir(directory)
df=pd.read_csv(file, delimiter=',')
#
node_map=df[['x','y','z']].copy()
#round up values in node map to prevent floating point errors
rounded_nodes=node_map.round(decimals=10)
#
##sort the nodes in ascending order
sorted_nodes=rounded_nodes.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)
sorted_data=df.round({'x':10,'y':10,'z':10})
sorted_data=sorted_data.sort_values(['x','y','z'],ascending=[True,True,True]).reset_index(drop=True)

thickness=1e-7*(length)
#sorted_data=sorted_data[(sorted_data['z']>4.0e-6) & (sorted_data['z']<(4.0e-6+thickness))].reset_index(drop=True)
#sorted_data=sorted_data[(sorted_data['z']>4e-6) & (sorted_data['z']<4.1e-6)].reset_index(drop=True)


if sorted_data['Ec'].min()<0:   
    sorted_data['Ec']=sorted_data['Ec']-sorted_data['Ec'].min()
    
#create dataframes for each xyz dimension in the mesh. this creates a dimension list 
#that gives us the total no. of grid points in any given direction
unique_x=sorted_data['x'].unique()
unique_y=sorted_data['y'].unique()
unique_z=sorted_data['z'].unique()


#sort these dataframes in acsending order
xvalues=pd.DataFrame(unique_x).sort_values([0],ascending=True).reset_index(drop=True)
yvalues=pd.DataFrame(unique_y).sort_values([0],ascending=True).reset_index(drop=True)
zvalues=pd.DataFrame(unique_z).sort_values([0],ascending=True).reset_index(drop=True)

sorted_data['Bg']=sorted_data['Landscape_Electrons']-sorted_data['Landscape_Holes']
Bg_dat=sorted_data[['Node','Bg']]
Bg_dat=Bg_dat.set_index('Node')
Bg_dat=Bg_dat.sort_index()
yn=abs(3.437-Bg_dat)
Uo=yn.round(4)
num_nodes=len(yn)


Name="Offset.out"
if os.path.isfile(Name):
    print ("File exist")
    os.remove(Name)
else:
    print ("File not exist")
f=open(Name, "a")
f.write('$MeshFormat\n\
2 0 8\n\
$NodeData\n\
1\n\
Offset_check\n\
1\n\
1.0\n\
3\n\
0\n\
1\n\
'+str(num_nodes)+'\n\
')

for i in range(num_nodes):
    f.write('           '+str(i)+'   '+str(Uo['Bg'].iloc[i])+'\n')
#    print(Uo['Bg'].iloc[i])
f.close()