# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:32:07 2022

@author: me_hi
"""

from functions import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import simplejson as json
from matplotlib import cm

directory = 'D:\\Research\\Simulation Data\\p_type_AlGaN_0.14_40nm_2\\Bias26'
file='p_type_AlGaN_0.14_40nm_2-out.vg_0.00.vd_0.00.vs_0.00.unified'

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

#arr_of_surf=[bottom_surf,top_surf,s1_surf,s2_surf,s3_surf,s4_surf]
arr_of_surf=[s1_surf]
#listofz=np.array(zvalues.index.tolist())
#
#listofz=listofz[listofz%10==0]

def plotsurf(surf,Var):

    zmap=surf[[surf.columns[1],surf.columns[2], Var ]].reset_index().round({'x':10,'y':10,'z':10})
    
    
    x=zmap[zmap.columns[1]].values
    
    y=zmap[zmap.columns[2]].values
    
    z=zmap[Var].values
    
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    
    Ec_array = np.empty(x_vals.shape + y_vals.shape)
    
    Ec_array.fill(np.nan)
    
    Ec_array[x_idx, y_idx] = zmap[Var].values
    
    cmap=cm.viridis
    
    
#    fig = plt.figure(figsize=(len(y_vals)/10, len(x_vals)/10))
    if len(y_vals)==len(x_vals):
            fig = plt.figure()
    else:
       fig = plt.figure(figsize=(len(y_vals)/20, len(x_vals)/20)) 


    plt.contourf(y_vals/10e-7,x_vals/10e-7,Ec_array,500,cmap=cmap) 

    plt.colorbar(orientation='horizontal')
    plt.axis('off')
        
    
#    
#p=plotsurf(cross_section,'Ev')
var='Comp'


def  volume_slicer(axes,var):
    val_array=sorted_data[['x','y','z',var]].copy()
    x=val_array['x'].values
    #
    y=val_array['y'].values
    
    z=val_array['z'].values
    
    val=val_array[var].values
    
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    z_vals, z_idx = np.unique(z, return_inverse=True)
    
    xx,yy,zz= np.meshgrid(x_vals, y_vals, z_vals, sparse=True)
    
    
    Ec_array = np.empty(x_vals.shape + y_vals.shape+z_vals.shape)
    Ec_array.fill(np.nan)
    Ec_array[x_idx, y_idx,z_idx] = val
    cmap=cm.viridis
    volume_slice(Ec_array, plane_orientation=axes,colormap='viridis')

#volume_slicer('z_axes','Ev')
def volumeplot(sorted_data):   

    val_array=sorted_data[['x','y','z',var]].copy()
    x=val_array['x'].values
    #
    y=val_array['y'].values
    
    z=val_array['z'].values
    
    val=val_array[var].values
    
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    z_vals, z_idx = np.unique(z, return_inverse=True)
    
    xx,yy,zz= np.meshgrid(x_vals, y_vals, z_vals, sparse=True)
    
    values = np.linspace(0., 1., 256)
    Ec_array = np.empty(x_vals.shape + y_vals.shape+z_vals.shape)
    Ec_array.fill(np.nan)
    Ec_array[x_idx, y_idx,z_idx] = val
    cmap=cm.viridis
    volume= pipeline.volume(pipeline.scalar_field(Ec_array),vmin=Ec_array.min(),vmax=Ec_array.max())
    #c = ctf.save_ctfs(volume._volume_property)
    #c['rgb']=cm.get_cmap('viridis')(values.copy())
    #ctf.load_ctfs(c, volume._volume_property)
    #volume.update_ctf = True

def returnmat(surf_type):
    surf=surf_type
    factor=5
    zmap=surf[[surf.columns[1],surf.columns[2], var ]].reset_index().round({'x':10,'y':10,'z':10})
    
    
    x=zmap[zmap.columns[1]].values
    
    y=zmap[zmap.columns[2]].values
    
    z=zmap[var].values
    
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    
    
    X,Y=np.meshgrid(x_vals,y_vals)
    
    Ec_array = np.empty(y_vals.shape + x_vals.shape)
    
    Ec_array.fill(np.nan)
    
    Ec_array[y_idx, x_idx] = zmap[var].values
    
    return x_vals, y_vals, X, Y, Ec_array
    
x_vals, y_vals, X, Y, Ec_array=returnmat(cross_section)
cmap=cm.viridis

if len(y_vals)==len(x_vals):
        fig = plt.figure()
else:
    fig = plt.figure(figsize=(len(x_vals)/5, len(y_vals)/5)) 
CS=plt.contourf(x_vals/1e-7,y_vals/1e-7,Ec_array-Ec_array.min(), 30, cmap=cm.viridis)
cbar=plt.colorbar()

# plt.clim(0,0.7)
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.tight_layout()

# fig=plt.figure()
# ax = fig.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X/1e-7, Y/1e-7, Ec_array-Ec_array.min(), cmap=cm.viridis, linewidth=0)
# #     fig.colorbar(surf)
# # colo
# # ax.set_zlim(0,0.7)
# ax.set_xlabel('x(nm)')
# ax.set_ylabel('y(nm)')
# # ax.set_zscale('log')
# ax.set_zlabel('Landscape Energy fluctuations')
# plt.tight_layout()