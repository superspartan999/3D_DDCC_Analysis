# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:27:40 2019

@author: Clayton
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import os


def extract_slice(data, slice_var, slice_val, drop=False):

    """
    This function grabs a 2D slice of a 3D data set. The function can set the
    variable and value as an argument.
    """

    if type(data) is not pd.DataFrame or type(slice_var) is not str:
        print('Input parameters of incorrect type.')
        return

    print("Slicing data...")
    my_filter = data[slice_var] == slice_val
    slice_data = data[my_filter]

    if drop:
        slice_data = slice_data.drop(slice_var, axis=1)

    return slice_data


def electric_field_z(df1, Ecom):
    #find all the values of z and put them in a list
    zvalues = df1['z'].unique()
    cols={}
    #create dataframe for conduction band and valence band
    Evalues=pd.DataFrame(columns=['z',Ecom])

    i=0
    #loop through different z values along the device
    for z in zvalues:
        #extract x-y plane for a z value
        zslice=extract_slice(df1,'z',z, drop=True)

        
        #average
        averagezsliceE=zslice[Ecom].mean()
        print averagezsliceE
        d1={'z':z,Ecom:averagezsliceE}
        Evalues.loc[i]=d1
        i=i+1


    return Evalues

def band_diagram_z(df1):
    #find all the values of z and put them in a list
    zvalues = df1['z'].unique()
    cols={}
    #create dataframe for conduction band and valence band
    Ecvalues=pd.DataFrame(columns=['z','Ec'])
    Evvalues=pd.DataFrame(columns=['z','Ev'])
    i=0
    #loop through different z values along the device
    for z in zvalues:
        #extract x-y plane for a z value
        zslice=extract_slice(df1,'z',z, drop=True)
        
        #average
        averagezsliceEc=zslice['Ec'].mean()
        averagezsliceEv=zslice['Ev'].mean()
        d1={'z':z,'Ec':averagezsliceEc}
        d2={'z':z,'Ev':averagezsliceEv}
        Ecvalues.loc[i]=d1
        Evvalues.loc[i]=d2
        i=i+1


    return Ecvalues,Evvalues 

def lowestpoint(df1):
    #find all the values of z and put them in a list
    zvalues = df1['z'].unique()
    cols={}
    #create dataframe for conduction band and valence band
   lvalues=pd.DataFrame(columns=['x','y','z','Ec'])

    i=0
    #loop through different z values along the device
    for z in zvalues:
        #extract x-y plane for a z value
        zslice=extract_slice(df1,'z',z, drop=True)
        l=zslice['Ec'].min()


    return Ecvalues,Evvalues 

#def jp_z(df1):
#    #find all the values of z and put them in a list
#    zvalues = df1['z'].unique()
#    cols={}
#    #create dataframe for conduction band and valence band
#    Ecvalues=pd.DataFrame(columns=['z','Ec'])
#    Evvalues=pd.DataFrame(columns=['z','Ev'])
#    i=0
#    #loop through different z values along the device
#    for z in zvalues:
#        #extract x-y plane for a z value
#        zslice=extract_slice(df1,'z',z, drop=True)
#        
#        #average
#        averagezsliceEc=zslice['Ec'].mean()
#        averagezsliceEv=zslice['Ev'].mean()
#        d1={'z':z,'Ec':averagezsliceEc}
#        d2={'z':z,'Ev':averagezsliceEv}
#        Ecvalues.loc[i]=d1
#        Evvalues.loc[i]=d2
#        i=i+1
#directory='D:\\HoletransportAlGaN_0.17_30nm_2'

#file='unified electric field data.csv'

directory = 'C:\\Users\\Clayton\\Desktop\\Older\\Bias8'

file = 'p_structure_0.17_10nm-out.vg_0.00.vd_3.50.vs_0.00.unified'
#
directory ='C:\\Users\\Clayton\\Desktop\\10nmAlGaN\\Bias8'
file = 'p_structure_0.17_10nm-out.vg_0.00.vd_-0.20.vs_0.00.unified'


#directory="C:\\Users\\Clayton\\Desktop\\CNSI test"
#file='p_structure_0.17_10nm-out.vg_0.00.vd_-0.20.vs_0.00.unified'
##
#directory ='C:\\Users\\Clayton\\Desktop\\10nmAlGaN\\Bias10'
#file= 'p_structure_0.17_10nm-out.vg_0.00.vd_0.00.vs_0.00.unified'

#directory ='C:\\Users\\Clayton\\Desktop\\30nmAlGaN\\Bias8'
#file= 'p_structure_0.17_30nm-out.vg_0.00.vd_-0.20.vs_0.00.unified'

os.chdir(directory)
#df=pd.read_csv(file, delimiter=',')
df=pd.read_csv(file, delimiter=',')
#
#Ecomponent='E'
#
#EcEv=band_diagram_z(df)
#
#Ec=EcEv[0]
#Ev=EcEv[1]
#
#Elfield=electric_field_z(df, Ecomponent)
#Elfield.plot('z',Ecomponent)
#plt.plot(Ec['z'],Ec['Ec'])
#plt.plot(Ev['z'],Ev['Ev'])
#plt.plot(Ec['z'], Ec['Ec'])

#df.plot('z','E')
#
#plt.scatter(df['z'],df['E'])
