# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:14:54 2018

@author: Clayton

"""
from __future__ import division
import pandas as pd

import scipy
import numpy as np

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#"Enter Input File of the structure"
StructureFile = "p_structure.csv";
Data = pd.read_csv(StructureFile);
NameJob = "p_structure_0.17_10nm";
 #=====================================================================================#

#(*Enter Coefficient of the structure*)

AugerCoefficient = "6.00000E-31";
RecombinationCoefficient = "2.00E-11";
 #=====================================================================================#


#(*Enter Filepath directory Server & File Name*)

FilePathServer = "/home/Clayton/Files/HoletransportAlGaN_0.17_10nm_2/";

#(*Input name*)
InputNameSh = "p_structure_0.17_10nm_";
InputName =[ NameJob+".inp"];

#(*Enter wanted Bias range *)

StartBias = -5;
EndBias = 5;
DeltaBias = 0.5;


#(**************************************************************)
#(*Number of point per Job *)
JobsPerFile = 2;
#(**************************************************************)

#(*Table of parameters *)
#converts dataframe column to list
Doping=np.array([x for pair in zip((Data['Doping'].tolist()),(Data['Doping'].tolist())) for x in pair])
Activation=np.array([x for pair in zip((Data['Activation'].tolist()),(Data['Activation'].tolist())) for x in pair])
Impurity=np.array([x for pair in zip((Data['Impurity'].tolist()),(Data['Impurity'].tolist())) for x in pair])
Emobility=np.array([x for pair in zip((Data['Mobilitye-'].tolist()),(Data['Mobilitye-'].tolist())) for x in pair])
Hmobility=np.array([x for pair in zip((Data['mobilityh+'].tolist()),(Data['mobilityh+'].tolist())) for x in pair])
ENonRad=Activation=np.array([x for pair in zip((Data['NonRade-'].tolist()),(Data['NonRade-'].tolist())) for x in pair])
HNonRad=np.array([x for pair in zip((Data['NonRadh+'].tolist()),(Data['NonRadh+'].tolist())) for x in pair])


#(*Calculation number files needed*)
NbPointBias = (EndBias - StartBias)/DeltaBias;
FilesNumber = NbPointBias/JobsPerFile;
Biases=np.arange(StartBias, EndBias+DeltaBias, DeltaBias)
BiasChunks=np.array_split(Biases,FilesNumber)
