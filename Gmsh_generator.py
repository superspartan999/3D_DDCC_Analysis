# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:03:18 2018

@author: Clayton
"""
import pandas as pd
from __future__ import division
import scipy

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

def FunctionMeshAlphaTest(Length, MeshMin, Prog) :

   for b in range(21):
       for i in range(1,b):
           bpair = Round[FindRoot[{Sum[MeshMin*Prog^(i - 1)] == Length} , {b, 20}][[1, 2]]];
       Progression = 
       SetPrecision[
      FindRoot[{Sum[MeshMin*b^(i - 1), {i, 1, Nbpair}] == 
         Length} , {b, 1.1}], 3][[1, 2]];
   {Nbpair, Progression}
   };