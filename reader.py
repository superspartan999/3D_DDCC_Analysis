# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:27:40 2019

@author: Clayton
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp


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

<<<<<<< HEAD
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

def testE(Ec):
    
    Evalues=pd.DataFrame(columns=['z','E'])
    
    for i in range(1,len(Ec)-1):
        E=(Ec.iloc[i+1]['Ec']-Ec.iloc[i-1]['Ec'])/(Ec.iloc[i+1]['z']-Ec.iloc[i-1]['z'])
        d={'z':Ec.iloc[i]['z'],'E':E}


        Evalues.loc[i]=d
        
        
    return Evalues

def NDAE(df):
    
    Evalues=pd.DataFrame(columns=['z','E'])
    
    
    
df=pd.read_csv('C:\\Users\\Clayton\\Desktop\\10nmAlGaN\\Bias8\\p_structure_0.17_10nm-out.vg_0.00.vd_-0.20.vs_0.00.unified', delimiter=' ')
df=df.drop(['Unnamed: 0'], axis=1)
#Ecomponent='Ez'

#EcEv=band_diagram_z(df)
#
#Ec=EcEv[0]
#Ev=EcEv[1]
#
#E=testE(Ec)
#
#E_z=electric_field_z(df, Ecomponent)
#
##plt.plot(E['z'],E['E'])
#plt.plot(E_z['z'],E_z['Ez'])


=======
mydf=pd.read_csv('C:\\Users\\Clayton\\Desktop\\10nmAlGaN\\Bias8\\p_structure_0.17_10nm-out.vg_0.00.vd_-0.20.vs_0.00.unified', delimiter=',')
Ecomponent='E'

E_z=electric_field_z(df, Ecomponent)

plt.plot(E_z['z'],E_z[Ecomponent])
>>>>>>> parent of d234a39... redefine points for easier reading
