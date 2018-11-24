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
#\
#
#(* Functions used in the algorithm *)
#(*****************************************************************************************************************************************)"

def FunctionBumpMesh(Length, MeshMin, Prog) :
#
    def y(a,r,n):
        f=a*(1-r**n)/(1-r)-Length
        return f
        
    f=lambda x:y(MeshMin,Prog,x)
    n=optimize.root(f,20)
    g=lambda r:y(MeshMin, r, round(n['x'][0]))
    prog=optimize.root(g,1.1)
    
    return round(n['x']), prog['x'][0]

sol= FunctionMeshAlphaTest(DeviceLength, MinMesh, ProgMesh)
    
def FunctionRegMesh(Length, MeshMin):
    
    return math.ceil(Length/MeshMin)

sol2=FunctionRegMesh(DeviceLength, MinMesh)
    
#           
   
