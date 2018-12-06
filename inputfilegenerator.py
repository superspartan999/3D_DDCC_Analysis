# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:14:54 2018

@author: Clayton

"""
from __future__ import division
import pandas as pd
import numpy as np

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#"Enter Input File of the structure"
StructureFile = "p_structure.csv";
Data = pd.read_csv(StructureFile,  converters={'mobilityh+': lambda x: str(x),'Mobilitye-': lambda x: str(x),'Activation': lambda x: str(x),'Doping': lambda x: str(x),'Impurity': lambda x: str(x),'NonRade-': lambda x: str(x),'NonRadh+': lambda x: str(x)});
NameJob = "p_structure_0.17_10nm";

#Read Width of structure. Assumes structure plane is a 2D square
DeviceWidth=(Data['L'][0])*(10**-7)
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
#converts dataframe column to list. Each layer is doubled to fit the mesh
Doping=np.array([x for pair in zip((Data['Doping'].tolist()),(Data['Doping'].tolist())) for x in pair])
Activation=np.array([x for pair in zip((Data['Activation'].tolist()),(Data['Activation'].tolist())) for x in pair])
Impurity=np.array([x for pair in zip((Data['Impurity'].tolist()),(Data['Impurity'].tolist())) for x in pair])
Emobility=np.array([x for pair in zip((Data['Mobilitye-'].tolist()),(Data['Mobilitye-'].tolist())) for x in pair])
Hmobility=np.array([x for pair in zip((Data['mobilityh+'].tolist()),(Data['mobilityh+'].tolist())) for x in pair])
ENonRad=Activation=np.array([x for pair in zip((Data['NonRade-'].tolist()),(Data['NonRade-'].tolist())) for x in pair])
HNonRad=np.array([x for pair in zip((Data['NonRadh+'].tolist()),(Data['NonRadh+'].tolist())) for x in pair])
Numoflayer=len(Doping)

#full dataframe for structure (including double layer)
StructureDF=pd.DataFrame({'Doping':Doping, 'Activation': Activation, 'Impurity': Impurity, 'Emobility': Emobility , 'Hmobility':Hmobility, 'ENonRad':ENonRad, 'HNonRad':HNonRad })


#(*Calculation number files needed*)
NbPointBias = (EndBias - StartBias)/DeltaBias;
FilesNumber = NbPointBias/JobsPerFile;
Biases=np.arange(StartBias, EndBias+DeltaBias, DeltaBias)
BiasChunks=np.array_split(Biases,FilesNumber)

for x in range(int(FilesNumber)):
    InputName = NameJob+"_IV_"+str(x+1)+".inp"
    f = open(InputName, "a")
    f.write("GeneratedbyPython\n\
ClaytonQwah2018\n\
\n\
$geninpbymatlab\n\
\n\
$gencompositionmap\n\
\n\
$ifelectrode\n\
0.0   0.0   0.01\n\
"+str(BiasChunks[x][0])+"   "+str(BiasChunks[x][len(BiasChunks[x])-1])+"   "+str(DeltaBias)+"\n\
0.0   0.0   0.01 \n\
0.0   0.0   0.01 \n\
$schottkyba\n\
0.0\n\
\n\
$precision\n\
1.00e-5\n\
\n\
$usedynaohmic\n\
\n\
$maxsteps\n\
1000\n\
\n\
$RoomT\n\
300.0\n\
\n\
$gmshfile\n\
"+str(NameJob)+".msh\n\
3\n\
\n\
! ----------------------\n\
$useperiodicboun\n\
LED_PBC.txt\n\
! ----------------------\n\
\n\
$outfile\n\
"+str(NameJob)+"-out\n\
\n\
\n\
**************************************\n\
\n\
$callstrainsolver\n\
"+str(InputName)+"\n\
\n\
$ifusepolfromstrain\n\
\n\
$usecgsolver\n\
\n\
$meshfilename\n\
"+str(NameJob)+".msh\n\
3\n\
\n\
$uselocalelement\n\
\n\
$layernumber\n\
"+str(Numoflayer)+"\n\
\n\
$fixedsurface\n\
1\n\
 6\n\
\n\
$type\n\
Wurzite\n\
\n\
$latticeconstant\n")
    for layer in range(0,Numoflayer):
        f.write("3.18900E+00   5.18500E+00\n")
    f.write("\n\
$elasticconstant\n\
\n")
    for layer in range(0,Numoflayer):
        f.write("3.67000E+01  1.35000E+01  1.03000E+01  4.05000E+01  9.50000E+00  1.\
23000E+0\n")
    f.write("\n\
$piezoelectric\n\
\n")
    for layer in range(0,Numoflayer):
        f.write("7.30000E-01  -4.90000E-01  -4.00000E-01\n")
    f.write("\n\
$physicalgroup\n")
    for layer in range(0,Numoflayer):
        f.write(str(layer+1)+"\n")
    f.write("\n\
$substratelattice\n\
3.18900 E + 00   5.18500 E + 00\n\
\n\
\n\
$PBCpoint_x\n\
0 "+str(DeviceWidth)+" e - 07\n\
\n\
$PBCpoint_y\n\
0 "+str(DeviceWidth)+" e - 07\n\
***************************************\n\
\n\
$ElBoundary\n\
2\n\
6 4 0.0000\n\
"+str(20+(Numoflayer-1)*18)+" 3 0.0000\n\
\n\
\n\
"+str(Numoflayer-1)+"\n")
    for layer in range(0,Numoflayer):
        f.write(str(layer+1)+" 0.0000 0.0000\n")
    f.write("volumenum(i),Eg(i),Ecoff(i),ep(i),charges(i),ND(i),Ea(i),impurity(i),\
meper(i),mepar(i),mlh(i),mhh(i),polz(i),poly(i),polx(i),mun(i),mup(i),\
taun(i),taup(i),rad(i)\n\
\n\
$Electriccoe\n\
"+str(Numoflayer)+"\n")
    for layer in range(0,Numoflayer):
       f.write(str(layer+1)+" 3.4370 0.6300 10.400 0.00000E+00 "+str(Doping[layer])+" \
"+str(Activation[layer])+" "+str(Impurity[layer])+"\
0.0000E-01 0.0000E-01 0.0000E+00 0.00000E+00 0.00000E+00 0.00E+00 0.00E+00 "+str(Emobility[layer])+ " "+str(Hmobility[layer])+" \
"+str(ENonRad[layer])+" "+str(HNonRad[layer])+" \
"+str(RecombinationCoefficient)+"\n")
       
    f.write("$ifdirectrecombine\n\
\n\
\n\
\n\
$ifschockley\n\
\n\
\n\
$ifsolveddn\n\
\n\
$UseAuger\n")
    for layer in range(0,Numoflayer):
        f.write(str(AugerCoefficient)+"\n")

    f.write("$landscape3D\n\
\n\
\n\
$landscapeDOS\n\
\n\
\n\
\n\
$assignparbyfunc\n\
\n\
\n\
\n\
\n\
\n\
\n\
$ifoutputall\n\
\n\
$outsetting\n\
\n\
$ifxscaled\n\
1.000E+00\n\
\n\
$ifyscaled\n\
1.000E+00\n\
\n\
$ifzscaled\n\
1.000E+00\n\
\n\
")                                               

         
f.close()
 

    
