# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:03:18 2018

@author: Clayton
"""
import pandas as pd

"Enter Input File of the structure"
StructureFile = "p_structure.csv";

"Enter Name of .geo file*"
NameJob = "p_structure_0.17_10nm";
Name = [NameJob+".geo"];

"Device Length"
DeviceLength=30

"Lateral mesh size in nm"
LateralMesh = 0.6;

"Progression parameter of the progressive mesh"
ProgMesh = 1.08;

"Minimum size of a mesh element"
MinMesh = 0.1;

"Import the structure from text file"
Data = pd.read_csv(StructureFile);
#DeviceLength = Data[[1, 2]];
print("Lateral Length of the structure ",DeviceLength, " nm")