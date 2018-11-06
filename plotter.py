# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:24:07 2018

@author: Clayton
"""
from matplotlib.pyplot import *
#function to plot band diagram
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

def electric_field_z(sorted_data):
    sorted_data['E']=E
        #find all the values of z and put them in a list
    zvalues = sorted_data['z'].unique()
    cols={}
    #create dataframe for conduction band and valence band
    Evalues=pd.DataFrame(columns=['z','E'])
    
    i=0
    #loop through different z values along the device
    for z in zvalues:
        #extract x-y plane for a z value
        zslice=extract_slice(sorted_data,'z',z, drop=True)
        
        #average
        averagezsliceE=zslice['E'].mean()
        d1={'z':z,'E':averagezsliceE}
        Evalues.loc[i]=d1
        i=i+1
    
    return Evalues


#
#
El=electric_field_z(sorted_data)
bd=band_diagram_z(sorted_data)    

axes = plt.gca()
axes.set_xlabel('z(cm)')
axes.set_ylabel('E(V/cm)')
plot(El['z'], El['E'])
plot(bd[0]['z'],bd[0]['Ec'])
plot(bd[1]['z'],bd[1]['Ev'])
