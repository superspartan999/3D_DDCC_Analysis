# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:03:18 2018

@author: Clayton
"""
from __future__ import division
import pandas as pd

import scipy as scp
import math
from scipy import optimize

#"Enter Input File of the structure"
StructureFile = "p_structure.csv";

#"Enter Name of .geo file*"
NameJob = "p_structure_0.17_10nm";
Name = [NameJob+".geo"];

#"Device Length"
DeviceLength=30

#"Lateral mesh size in nm"
LateralMesh = 0.6;

#"Progression parameter of the progressive mesh"
ProgMesh = 1.08;

#"Minimum size of a mesh element"
MinMesh = 0.1;

#"Import the structure from text file"
Data = pd.read_csv(StructureFile);

#DeviceLength = Data[[1, 2]];
print("Lateral Length of the structure ",Data.at[0,'L'], " nm")  


    
 #(*****************************************************************************************************************************************)
#(* Functions used in the algorithm *)
#(*****************************************************************************************************************************************)"

#function to create a bump in the mesh
def FunctionProgMesh(Length, MeshMin, Prog) :
#
    def y(a,r,n):
        f=a*(1-r**n)/(1-r)-Length
        return f
        
    f=lambda x:y(MeshMin,Prog,x)
    n=optimize.root(f,20)
    g=lambda r:y(MeshMin, r, round(n['x'][0]))
    prog=optimize.root(g,1.1)
    
    return round(n['x'][0]), prog['x'][0], 1/prog['x'][0] 

sol= FunctionProgMesh(DeviceLength, MinMesh, ProgMesh)

#function for regular mesh    
def FunctionRegMesh(Length, MeshMin):
    
    f= math.ceil(Length/MeshMin)
    return f
sol2=FunctionRegMesh(DeviceLength, MinMesh)

#function to output table with mesh parameters for each layer. Duplicate each layer and do a double progression mesh to create a bump in the layer
def MeshConstructor(Data):
    Data["MeshParam"]='s'  
    
      #duplicate each layer
    Data = Data.loc[Data.index.repeat(2)]
    
    #reset index
    Data = Data.reset_index(drop=True)
    
    #dividing the thickness of each layer to reset the thickness
    Data['nm']= Data['nm']/2
    
    for i in range(0,len(Data),2):
        
    
    
    
        if Data.iloc[i]['type'] in ['QW', 'CAP']:
            meshparam=FunctionRegMesh(Data.iloc[i]['nm'],MinMesh)
            Data.at[i,"MeshParam"]=meshparam
            Data.at[i+1,"MeshParam"]=meshparam
    
        else:
            meshparam=(FunctionProgMesh(Data.iloc[i]['nm'],MinMesh,ProgMesh)[0],\
                       FunctionProgMesh(Data.iloc[i]['nm']/2,MinMesh,ProgMesh)[1],FunctionProgMesh(Data.iloc[i]['nm']/2,MinMesh,ProgMesh)[2])
    
            Data.at[i,"MeshParam"]=(meshparam[0],meshparam[1])
            Data.at[i+1,"MeshParam"]=(meshparam[0],meshparam[2])
    
    return Data

Data=MeshConstructor(Data)


        
            
    
#           
   
