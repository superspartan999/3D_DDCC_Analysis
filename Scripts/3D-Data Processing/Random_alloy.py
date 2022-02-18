# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 00:59:21 2022

@author: me_hi
"""

import numpy as np
import random as rd

import os
import pandas as pd
from functions import *

import simplejson as json
from matplotlib import cm

# x=np.linspace(0,100,1000)
# y=np.linspace(0,100,1000)
# z=np.linspace(0,100,1000)


# def generate_point(mean_x, mean_y, deviation_x, deviation_y):
#     return rd.gauss(mean_x, deviation_x), rd.gauss(mean_y, deviation_y)

# cluster_mean_x = 100
# cluster_mean_y = 100
# cluster_deviation_x = 50
# cluster_deviation_y = 50
# point_deviation_x = 5
# point_deviation_y = 5

# number_of_clusters = 5
# points_per_cluster = 50


# cluster_centers = [generate_point(cluster_mean_x,
#                                   cluster_mean_y,
#                                   cluster_deviation_x,
#                                   cluster_deviation_y)
#                    for i in range(number_of_clusters)]


# points = [generate_point(center_x,
#                          center_y,
#                          point_deviation_x,
#                          point_deviation_y)
#           for center_x, center_y in cluster_centers
#           for i in range(points_per_cluster)]

# directory = 'D:\\Research\\Simulation Data\\p_type_InGaN_0.10_30nm_2\\Bias10'
# file = 'p_type_InGaN_0.10_30nm_2-out.vg_0.00.vd_0.00.vs_0.00.unified'

material='InGaN'
comp='0.4'
directory = 'C:\\Users\\me_hi\\Downloads\\Research\\'+material+'_M1com'+comp
file=material+'_M1com'+comp+'-out.vg_    0.000.vd_    0.000.vs_    0.000.unified'
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

#sorted_data=sorted_data[(sorted_data['z']>3e-6) & (sorted_data['z']<7e-6)]
#sorted_data=sorted_data[sorted_data['z']<5.6e-6]

#create dataframes for each xyz dimension in the mesh. this creates a dimension list 
#that gives us the total no. of grid points in any given direction
unique_x=sorted_data['x'].unique()
unique_y=sorted_data['y'].unique()
unique_z=sorted_data['z'].unique()


#sort these dataframes in acsending order
xvalues=pd.DataFrame(unique_x).sort_values([0],ascending=True).reset_index(drop=True)
yvalues=pd.DataFrame(unique_y).sort_values([0],ascending=True).reset_index(drop=True)
zvalues=pd.DataFrame(unique_z).sort_values([0],ascending=True).reset_index(drop=True)

bottom_surf=extract_slice(sorted_data,'z',zvalues.iloc[0][0],drop=True)
top_surf=extract_slice(sorted_data,'z',zvalues.iloc[-1][0],drop=True)
cross_section=extract_slice(sorted_data,'z',zvalues.iloc[int((len(zvalues)-1)/2)][0],drop=True)
p_section=extract_slice(sorted_data,'z',9e-6,drop=True)
s1_surf=extract_slice(sorted_data,'x',xvalues.iloc[0][0],drop=True)
s2_surf=extract_slice(sorted_data,'x',xvalues.iloc[-1][0],drop=True)
s3_surf=extract_slice(sorted_data,'y',yvalues.iloc[0][0],drop=True)
s4_surf=extract_slice(sorted_data,'y',yvalues.iloc[-1][0],drop=True)

var='Comp'
surf=cross_section


test_surf=cross_section


test_comp=np.linspace(float(comp)-0.05,float(comp)+0.05,100)
ratio_list=np.zeros(100)
diff_list=np.zeros(100)

for i,compo in enumerate(test_comp):
    
    yellow=0
    blue=0
    for index, row in test_surf.iterrows():
        if test_surf['Comp'].loc[index] > compo:
            # test_surf['Comp'].loc[index]=1
            yellow=yellow+1
        
        else:
            # test_surf['Comp'].loc[index]=0
            blue=blue+1
            
            
    ratio=yellow/(blue+yellow)
    ratio_list[i]=ratio
    diff_list[i]=ratio-float(comp)
yellow=0
blue=0
index_of_best_compo=np.argmin(abs(diff_list))
filter_comp=test_comp[index_of_best_compo]
for index, row in surf.iterrows():
    if surf['Comp'].loc[index] > filter_comp:
        surf['Comp'].loc[index]=1


    
    else:
        surf['Comp'].loc[index]=0
    
    


# cbar.set_ticks(ticks)
# cbar.ax.tick_params(labelsize=18)

factor=5
zmap=surf[[surf.columns[1],surf.columns[2], var ]].reset_index().round({'x':10,'y':10,'z':10})


x=zmap[zmap.columns[1]].values

y=zmap[zmap.columns[2]].values

z=zmap[var].values

x_vals, x_idx = np.unique(x, return_inverse=True)
y_vals, y_idx = np.unique(y, return_inverse=True)

X,Y=np.meshgrid(x_vals,y_vals)

Ec_array = np.empty(x_vals.shape + y_vals.shape)

Ec_array.fill(np.nan)

Ec_array[x_idx, y_idx] = zmap[var].values

cmap=cm.viridis

# if len(y_vals)==len(x_vals):
#         fig = plt.figure()
# else:
#     fig = plt.figure(figsize=(len(y_vals)/30, len(x_vals)/30)) 
CS=plt.contourf(y_vals/1e-7,x_vals/1e-7,Ec_array,100,cmap=cm.viridis) 



# cbar=plt.colorbar(orientation='horizontal')


random_sampler=np.zeros(len(zmap.reset_index()))

for index, row in zmap.reset_index().iterrows():
    
    n=rd.randint(0,2)
    if n <1:
            random_sampler[index]=index
            
random_sampler=random_sampler.nonzero()[0]     




newzmap=zmap.copy()

for index,row in newzmap.iterrows():
     if row['index'] not in random_sampler:
         newzmap['Comp']=np.nan
    

# comp_array = np.empty(x_vals.shape + y_vals.shape)

# comp_array.fill(np.nan)

# comp_array[x_idx, y_idx] = zmap[var].values

# for index, row in surf.iterrows():
    
#     if surf['index'] not in random_sampler:
#         comp_array
        
        
    
    
# for index in random_sampler:
#      surf['Comp'].loc[index]
# for index in random_sampler:

# composition_sim=yellow/(blue+yellow)
# ticks=np.linspace(surf[var].min()*100,surf[var].max()*100,5)


