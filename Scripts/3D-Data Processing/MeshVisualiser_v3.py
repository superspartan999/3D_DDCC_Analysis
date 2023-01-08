# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:45:24 2020

@author: Clayton
"""
from functions import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import scipy.interpolate as interp
from scipy.ndimage import gaussian_filter
# from tvtk.util import ctf
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from mpl_toolkits.mplot3d import axes3d, Axes3D 
# from math import floor, sqrt
# import networkx as nx
# from networkx.readwrite import json_graph
import simplejson as json
from matplotlib import cm
from mayavi.mlab import *
# from itertools import *
# import heapq
# from scipy.spatial import KDTree
# import scipy.ndimage as ndimage
# from mayavi.mlab import *
from mayavi import mlab
from matplotlib.colors import LightSource
# from matplotlib.mlab import griddata

#
#def make_frame(t):
#    
#    return frame_for_time_t
#
#animation=VideoClip(make_frame,duration=duration)

dirlist=['C:\\Users\\me_hi\\Downloads\\InGaN0.5nm','C:\\Users\\me_hi\\Downloads\\InGaN_newmod_polar_strain_1L_alloyth_M1com0.3_alloy30.0nm']

dirlist=['C:\\Users\\me_hi\\Downloads\\InGaN0.5nm']

for directory in dirlist:
# directory = 'C:\\Users\\me_hi\\Downloads\\InGaN0.5nm'
# directory = 'C:\\Users\\me_hi\\Downloads\\InGaN_newmod_polar_strain_1L_alloyth_M1com0.3_alloy30.0nm'
    os.chdir(directory)
    
    
    # file = 'InGaN_newmod_polar_strain_1L_alloyth_M1com0.3_alloy30.0nm_T300.0_1e16dop_N20200325_0.5nm_Vd_0_0_1-out.vg_0.000.vd_0.000.vs_0.000.unified'
    file = 'InGaN_newmod_polar_strain_1L_alloyth_M1com0.3_alloy30.0nm_T300.0_1e16dop_N20200325_Vd_0_0_1-out.vg_0.000.vd_0.000.vs_0.000.unified'
    
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
            
        
    #%%   
    #p=plotsurf(cross_section,'Ev')
    var='Landscape_Electrons'
    
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
    
    csurf=cross_section
    
    
    factor=5
    zmap=csurf[[csurf.columns[1],csurf.columns[2], var ]].reset_index().round({'x':10,'y':10,'z':10})
    
    # zmap=full_zmap[full_zmap['x']>0.5e-6] 
    # zmap=zmap[zmap['y']>0.5e-6] 
    # zmap=zmap[zmap['x']<2.5e-6] 
    # zmap=zmap[zmap['y']<2.5e-6] 
    
    x=zmap[zmap.columns[1]].values
    
    
    y=zmap[zmap.columns[2]].values
    
    z=zmap[var].values
    
    
    
    # x=x[x<2e-6]
    
    # y= y[y>1e-6]
    # y=y[y<2e-6]
    
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    
    
    X,Y=np.meshgrid(x_vals,y_vals)
    
    Ec_array = np.empty(x_vals.shape + y_vals.shape)
    
    Ec_array.fill(np.nan)
    
    Ec_array[x_idx, y_idx] = zmap[var].values
    
    Ec_array=Ec_array-np.mean(Ec_array)
    
    f=interp.interp2d(x_vals, y_vals, Ec_array, kind='cubic')
    
    scalefactor=10
    xnew = np.arange(x_vals[0], x_vals[-1], 1e-8)
    ynew = np.arange(y_vals[0], y_vals[-1], 1e-8)
    
    X_new,Y_new=np.meshgrid(xnew,ynew)
    Ec_new=f(xnew,ynew)
    # av=av_values(sorted_data,'Landscape_Electrons')
    # plt.plot(av['z']/1e-7,1000*(av['Landscape_Electrons']-np.mean(av['Landscape_Electrons'].values)))
    # # plt.plot(x_vals/1e-7,np.mean(Ec_array,axis=0)*1000)
    
    # plt.plot(xnew/1e-7, Ec_new[int(len(Ec_new)/2), :]*1000)
    # plt.plot(x_vals/1e-7, Ec_array[int(len(Ec_array)/2), :]*1000)
    # plt.f
    
    # freq, bins=np.histogram(sorted_data['Landscape_Electrons'],bins=50)
    # plt.bar(((bins[:-1])-np.mean(sorted_data['Landscape_Electrons']))*1000,freq/np.sum(freq),width=2)
    
    

# var='Landscape_Electrons'

# def  volume_slicer(axes,var):
#     val_array=sorted_data[['x','y','z',var]].copy()
#     x=val_array['x'].values
#     #
#     y=val_array['y'].values
    
#     z=val_array['z'].values
    
#     val=val_array[var].values
    
#     x_vals, x_idx = np.unique(x, return_inverse=True)
#     y_vals, y_idx = np.unique(y, return_inverse=True)
#     z_vals, z_idx = np.unique(z, return_inverse=True)
    
#     xx,yy,zz= np.meshgrid(x_vals, y_vals, z_vals, sparse=True)
    
    
#     Ec_array = np.empty(x_vals.shape + y_vals.shape+z_vals.shape)
#     Ec_array.fill(np.nan)
#     Ec_array[x_idx, y_idx,z_idx] = val
#     cmap=cm.viridis
#     volume_slice(Ec_array, plane_orientation=axes,colormap='viridis')

# #volume_slicer('z_axes','Ev')
# def volumeplot(sorted_data):   

#     val_array=sorted_data[['x','y','z',var]].copy()
#     x=val_array['x'].values
#     #
#     y=val_array['y'].values
    
#     z=val_array['z'].values
    
#     val=val_array[var].values
    
#     x_vals, x_idx = np.unique(x, return_inverse=True)
#     y_vals, y_idx = np.unique(y, return_inverse=True)
#     z_vals, z_idx = np.unique(z, return_inverse=True)
    
#     xx,yy,zz= np.meshgrid(x_vals, y_vals, z_vals, sparse=True)
    
#     values = np.linspace(0., 1., 256)
#     Ec_array = np.empty(x_vals.shape + y_vals.shape+z_vals.shape)
#     Ec_array.fill(np.nan)
#     Ec_array[x_idx, y_idx,z_idx] = val
#     cmap=cm.viridis
#     volume= pipeline.volume(pipeline.scalar_field(Ec_array),vmin=Ec_array.min(),vmax=Ec_array.max())
#     #c = ctf.save_ctfs(volume._volume_property)
#     #c['rgb']=cm.get_cmap('viridis')(values.copy())
#     #ctf.load_ctfs(c, volume._volume_property)
#     #volume.update_ctf = True

# csurf=cross_section


# factor=5
# zmap=csurf[[csurf.columns[1],csurf.columns[2], var ]].reset_index().round({'x':10,'y':10,'z':10})

# # zmap=full_zmap[full_zmap['x']>0.5e-6] 
# # zmap=zmap[zmap['y']>0.5e-6] 
# # zmap=zmap[zmap['x']<2.5e-6] 
# # zmap=zmap[zmap['y']<2.5e-6] 

# x=zmap[zmap.columns[1]].values


# y=zmap[zmap.columns[2]].values

# z=zmap[var].values



# # x=x[x<2e-6]

# # y= y[y>1e-6]
# # y=y[y<2e-6]

# x_vals, x_idx = np.unique(x, return_inverse=True)
# y_vals, y_idx = np.unique(y, return_inverse=True)


# X,Y=np.meshgrid(x_vals,y_vals)

# Ec_array = np.empty(x_vals.shape + y_vals.shape)

# Ec_array.fill(np.nan)

# Ec_array[x_idx, y_idx] = zmap[var].values

# Ec_array=Ec_array-np.mean(Ec_array)

# f=interp.interp2d(x_vals, y_vals, Ec_array, kind='cubic')

# scalefactor=10
# xnew = np.arange(x_vals[0], x_vals[-1], 1e-8)
# ynew = np.arange(y_vals[0], y_vals[-1], 1e-8)

# X_new,Y_new=np.meshgrid(xnew,ynew)
# Ec_new=f(xnew,ynew)

# plt.plot(xnew/1e-7, Ec_new[int(len(Ec_new)/2), :]*1000)

# import matplotlib as mpl

# mpl.rcParams['axes.linewidth'] = 2
# plane=np.ones(shape=[len(xnew),len(ynew)])
# plt.rc('font',family = 'arial',  size=20)
# plt.tick_params(width=3, length=6)
# plt.xlabel('x (nm)', family='Arial')
# # plt.grid()
# plt.tight_layout()
# # plt.ylabel(family='Arial')
# # plt.plot(xnew/1E-7,np.mean(Ec_new, axis=1))
# from matplotlib import ticker
# plt.figure()
# con=plt.contourf(xnew/1e-7, ynew/1e-7,Ec_new,100, cmap=cm.viridis)
# # plt.clim(vmin=-200 ,vmax=200) 
# # plt.tight_layout()
# cb=plt.colorbar(con)

# tick_locator = ticker.MaxNLocator(nbins=3)
# cb.locator=tick_locator
# cb.update_ticks()
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# X_new,Y_new=np.meshgrid(xnew,ynew)
# ls = LightSource(azdeg=0, altdeg=95)
# # shade data, creating an rgb array.
# rgb = ls.shade(Ec_new, plt.cm.RdYlBu)
# ls = LightSource(270, 45)
# # To use a custom hillshading mode, override the built-in shading and pass
# # in the rgb colors of the shaded surface calculated from "shade".
# rgb = ls.shade(Ec_new, cmap=cm.plasma, vert_exag=0.1, blend_mode='soft')
# surfplot = ax.plot_surface(X_new, Y_new, Ec_new, rstride=1, cstride=1, facecolors=rgb,
#                        linewidth=0, antialiased=False, shade=False)
# # fig.colorbar(rgb, shrink=0.5, aspect=5)
# ax.grid(False)
# ax.view_init(elev=0, azim=90)
# ax.get_yaxis().set_visible(False)
# ax.set_zlim(0.05,0.2)
# pts=points3d(zmap['x'],zmap['y'], zmap['Comp'], zmap['Comp'], scale_mode='none', scale_factor=0.2)
# mesh = pipeline.delaunay2d(pts)
# surf = pipeline.surface(mesh)
# # cmap=cm.viridis

# if len(y_vals)==len(x_vals):
#         fig = plt.figure()
# else:
#    fig = plt.figure(figsize=(len(y_vals)/30, len(x_vals)/30)) 
# CS=plt.contourf(y_vals/1e-7,x_vals/1e-7,Ec_array*100,100,cmap=cm.viridis) 
# cbar=plt.colorbar(orientation='vertical')
# ticks=np.linspace(surf[var].min()*100,surf[var].max()*100,5)
# cbar.set_ticks(ticks)
# cbar.ax.tick_params(labelsize=18)
#CS2=plt.contour(x_vals/1e-7,y_vals/1e-7,Ec_array, colors='black',linewidths=0.5)


#
#n=sorted_data['n'].values

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#x_vals_new=np.arange(x_vals.min(),x_vals.max(),np.diff(x_vals)[0]/factor)
#y_vals_new=np.arange(y_vals.min(),y_vals.max(),np.diff(x_vals)[0]/factor)
#Z = scp.interpolate.interp2d(x_vals, y_vals, Ec_array, kind='cubic')
#z_new=Z(x_vals_new,y_vals_new)
#X_new,Y_new=np.meshgrid(x_vals_new,y_vals_new)
##ls = LightSource(azdeg=0, altdeg=65)
##rgb = ls.shade(z_new, plt.cm.RdYlBu)
#ax.plot_surface(Y,X,Ec_array, cmap='viridis',antialiased=True)

#plotting outer surfaces
 
#zslice=extract_slice(sorted_data,'z',zvalues.iloc[int((len(zvalues)-1)/2)][0],drop=True)
#print(i)

#Var='Ec'
#zmap=surf[[surf.columns[1],surf.columns[2], Var ]].reset_index().round({'x':10,'y':10,'z':10})
#
#
#x=zmap[zmap.columns[1]].values
#
#y=zmap[zmap.columns[2]].values
#
#z=zmap[Var].values
#
#x_vals, x_idx = np.unique(x, return_inverse=True)
#y_vals, y_idx = np.unique(y, return_inverse=True)
#
#Ec_array = np.empty(x_vals.shape + y_vals.shape)
#
#Ec_array.fill(np.nan)
#
#Ec_array[x_idx, y_idx] = zmap[Var].values
#
#cmap=cm.viridis
#
#fig = plt.figure(figsize=(len(y_vals)/20, len(x_vals)/20))
#plt.contourf(y_vals/10e-7,x_vals/10e-7,Ec_array,100,cmap=cmap) 
#
#plt.colorbar()

#    
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#X = np.linspace(-5, 5, 43)
#Y = np.linspace(-5, 5, 28)
