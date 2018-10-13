# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:14:54 2018

@author: Clayton

"""
from __future__ import division
import pandas as pd

import scipy
import numpy as np


#"Enter Input File of the structure"
StructureFile = "p_structure.csv";
Data = pd.read_csv(StructureFile);
NameJob = "p_structure_0.17_10nm";
 #=====================================================================================#

#(*Enter Coefficient of the structure*)

AugerCoefficient = "6.00000E-31";
RecombinaisonCoefficient = "2.00E-11";
 #=====================================================================================#


#(*Enter Filepath directory Server & File Name*)

FilePathServer = "/home/Clayton/Files/HoletransportAlGaN_0.17_10nm_2/";

#(*Input name*)
InputNameSh = "p_structure_0.17_10nm_";
InputName =[ NameJob+".inp"];

#(*Enter wanted Bias range *)

StartBias = -4.0;
EndBias = 6.0;
DeltaBias = 0.5;


#(**************************************************************)
#(*Number of point per Job *)
JobsPerFile = 2;
#(**************************************************************)

#(*Table of paramaters *)
#converts dataframe column to list
Doping=np.array([x for pair in zip((Data['Doping'].tolist()),(Data['Doping'].tolist())) for x in pair])
Activation=np.array([x for pair in zip((Data['Activation'].tolist()),(Data['Activation'].tolist())) for x in pair])
#converts dataframe column to list
I=np.array(Data['Impurity'].tolist())
#duplicates element in the list
Impurity=np.array([x for pair in zip(I,I) for x in pair])
#converts dataframe column to list
Me=np.array(Data['Mobilitye-'].tolist())
#duplicates element in the list
Emobility=np.array([x for pair in zip(Me,Me) for x in pair])
#converts dataframe column to list
Mh=np.array(Data['mobilityh+'].tolist())
#duplicates element in the list
Hmobility=np.array([x for pair in zip(Mh,Mh) for x in pair])
#converts dataframe column to list




#(**************************************************************)

#(*Calculation number files needed*)
#NbPointBias = (EndBias - StartBias)/DeltaBias;
#FilesNumber = NbPointBias/JobsPerFile;
#PreBias = Table[{StartBias + DeltaBias + (i - 1)*(MaxPointPerFiles*DeltaBias), 
#    StartBias + i*(MaxPointPerFiles*DeltaBias)}, {i, 2, FilesNumber}];
#Bias = Prepend[PreBias, {StartBias, StartBias + (MaxPointPerFiles*DeltaBias)}];