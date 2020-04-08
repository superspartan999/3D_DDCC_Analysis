# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:14:54 2018

@author: Clayton

"""
from __future__ import division
import pandas as pd
import numpy as np
import os

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#"Enter Input File of the structure"
StructureFile = "p_structure.csv";
Data = pd.read_csv(StructureFile,  converters={'mobilityh+': lambda x: str(x),'Mobilitye-': lambda x: str(x),'Activation': lambda x: str(x),'Doping': lambda x: str(x),'Impurity': lambda x: str(x),'NonRade-': lambda x: str(x),'NonRadh+': lambda x: str(x)});
NameJob = "p_structure_0.17_10nm";

#Read Width of structure. Assumes structure plane is a 2D square
DeviceWidth=(Data['L'][0])
 #=====================================================================================#

#(*Enter Coefficient of the structure*)

AugerCoefficient = "6.00000E-31";
RecombinationCoefficient = "2.00E-11";
 #=====================================================================================#


#(*Enter Filepath directory Server & File Name*)

FilePathServer = "/home/Clayton/Files/HoletransportAlGaN_0.17_10nm_2/";

#(*Input name*)
InputNameSh = "p_structure_0.17_10nm_";

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

#=-----------------------------------------------------------------------------------------------------------------------#
#Writing the input file
#=-----------------------------------------------------------------------------------------------------------------------#
for x in range(int(FilesNumber)):
    #input file name
    InputName = NameJob+"_"+str(x+1)+".inp"
    
    #delete existing file to prevent overwriting
    os.remove(InputName)

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
\n\
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
")
    for layer in range(0,Numoflayer):
        f.write("3.67000E+01  1.35000E+01  1.03000E+01  4.05000E+01  9.50000E+00  1.\
23000E+0\n")
    f.write("\n\
$piezoelectric\n\
")
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
0 "+str(int(DeviceWidth))+"E-07\n\
\n\
$PBCpoint_y\n\
0 "+str(int(DeviceWidth))+"E-07\n\
  ***************************************\n\
\n\
$ElBoundary\n\
2\n\
6 4 0.0000\n\
"+str(20+(Numoflayer-1)*18)+" 3 0.0000\n\
\n\
\n\
"+str(Numoflayer)+"\n")
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
       
    f.write("\n\
\n\
\n\
$ifdirectrecombine\n\
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

#=-----------------------------------------------------------------------------------------------------------------------#
#writing the job file
#=-----------------------------------------------------------------------------------------------------------------------#
for x in range(int(FilesNumber)):
    JobName = NameJob+"_"+str(x+1)+".sh"
    os.remove(JobName)
    g = open(JobName, "a")
    g.write("#!/bin/bash\n\
#PBS - S /bin/bash\n\
#PBS - N "+str(NameJob)+"_"+str(x+1)+"\n\
#PBS - l nodes=1:ppn=8,mem=80gb,walltime=300:00:00,nice=15\n\
#PBS - q long\n\
#PBS - o "+str(FilePathServer)+"Bias"+str(x+1)+"\n\
#PBS - e "+str(FilePathServer)+"Bias"+str(x+1)+"\n\
cd "+str(FilePathServer)+"/Bias"+str(x+1)+"\n\
export MKL_NUM _THREADS=8\n\
export OPENMP_NUM _THREADS=8\n\
3D-ddcc-dyna.exe "+str(JobName)+" > test"+str(x+1)+".txt\n\
\n\
wait\n\
\n\
")
 
g.close()

#=-----------------------------------------------------------------------------------------------------------------------#
#writing the file to submit jobs
#=-----------------------------------------------------------------------------------------------------------------------#
FileName = NameJob+"_Jobs.sh"
os.remove(FileName)
h= open(FileName, "a")   
h.write(
"#!/bin/bash\n\
#PBS - S /bin/bash\n\
\n\
")
for x in range(int(FilesNumber)): 
    JobName = NameJob+str(x+1)+".sh"
    h.write(
"qsub "+JobName+"\n\
sleep 1\n\
")

h.close()  

#=-----------------------------------------------------------------------------------------------------------------------#
#writing the the Global file
#=-----------------------------------------------------------------------------------------------------------------------#   
GlobalFile=NameJob+"_Global.sh"
os.remove(GlobalFile)
i=open(GlobalFile, "a")
i.write(
"shopt - s extglob\n\
chmod 777 *\n\
gmsh - 3 "+str(NameJob)+".geo\n\
ifort PBC30.f90\n\
./a.out " +str(NameJob)+".msh\n\
./compile.bat\n\
./"+str(NameJob)+" _Copy.sh\n\
./"+str(NameJob)+" _Jobs.sh\n\
")    

i.close()



#=-----------------------------------------------------------------------------------------------------------------------#
#writing the file to create separate directories for each job 
#=-----------------------------------------------------------------------------------------------------------------------#

FileName = NameJob+"_Copy.sh"
os.remove(FileName)
j = open(FileName, "a")
j.write(
"#!/bin/bash\n\
#PBS - S /bin/bash\n\
\n\
")

for x in range(int(FilesNumber)): 
    JobName = NameJob+str(x+1)+".sh"
    j.write(
"mkdir Bias "+str(x+1)+"\n\
cp ./* ./Bias"+str(x+1)+"\n\
\n\
")

j.close()


#=-----------------------------------------------------------------------------------------------------------------------#
#writing the file to merge IV files from different folders
#=-----------------------------------------------------------------------------------------------------------------------#

FileName = "MergeIV.sh"
os.remove(FileName)
k = open(FileName, "a")

k.write(
"#!/bin/bash\n\
#PBS - S /bin/bash\n\
\n\
cat")
for x in range(int(FilesNumber)): 
    k.write(" "+str(FilePathServer)+"Bias"+str(x)+"/"+str(NameJob)+"-out.ivn")

k.close()  
  
  
