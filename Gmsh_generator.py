# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:03:18 2018

@author: Clayton
"""
from __future__ import division
import pandas as pd

import numpy as np
import os
import math
from scipy import optimize

#"Enter Input File of the structure"
StructureFile = "p_structure.csv";

#"Enter Name of .geo file*"
NameJob = "p_structure_0.17_10nm";
Name = NameJob+".geo";

os.remove(Name)

#"Device Length"
DeviceLength=30

#"Lateral mesh size in nm"
LateralMesh = 0.6;

#"Progression parameter of the progressive mesh"
ProgMesh = 1.08;

#"Minimum size of a mesh element"
MinMesh = 0.1;

#"Import the structure from text file"
Structure = pd.read_csv(StructureFile);

#DeviceLength = Data[[1, 2]];
print("Lateral Length of the structure ",Structure.at[0,'L'], " nm")  


    
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



#function for regular mesh    
def FunctionRegMesh(Length, MeshMin):
    
    f= math.ceil(Length/MeshMin)
    return f




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
        
        #if layer is a quantum well or quantum barrier, use a regular mesh
        if Data.iloc[i]['type'] in ['QW', 'CAP']:
            meshparam=FunctionRegMesh(Data.iloc[i]['nm'],MinMesh)
            Data.at[i,"MeshParam"]=meshparam
            Data.at[i+1,"MeshParam"]=meshparam
        #if any other layer, use a bump mesh    
        else:
            
            meshparam=(FunctionProgMesh(Data.iloc[i]['nm'],MinMesh,ProgMesh)[0],\
                       FunctionProgMesh(Data.iloc[i]['nm']/2,MinMesh,ProgMesh)[1],FunctionProgMesh(Data.iloc[i]['nm']/2,MinMesh,ProgMesh)[2])
    
            Data.at[i,"MeshParam"]=(round(meshparam[0],3),round(meshparam[1],3))
            Data.at[i+1,"MeshParam"]=(round(meshparam[0],3),round(meshparam[2],3))
    
    return Data

def Pointfunc(NVirt,row):
    for i in range(0,NVirt):
        temprow=np.array(['p'+str(i)+'1','p'+str(i)+'2','p'+str(i)+'3','p'+str(i)+'4'])
        row=np.vstack((row,temprow))
        
    return row

def Linefunc(NVirt,lines):
    for i in range(0,NVirt):
        temprow=np.array(['l'+str(i)+'1','l'+str(i)+'2','l'+str(i)+'3','l'+str(i)+'4','l'+str(i)+'5','l'+str(i)+'6','l'+str(i)+'7','l'+str(i)+'8'])
        lines=np.vstack((lines,temprow))
        
    return lines

def Lineloopfunc(NVirt,lines):
    for i in range(0,NVirt):
        temprow=np.array(['ll'+str(i)+'1','ll'+str(i)+'2','ll'+str(i)+'3','ll'+str(i)+'4','ll'+str(i)+'5'])
        lines=np.vstack((lines,temprow))
        #lines=np.transpose(lines)
        
    return lines

def Surfacefunc(NVirt,surf):
    for i in range(0,NVirt):
        temprow=np.array(['ps'+str(i)+'1','ps'+str(i)+'2','ps'+str(i)+'3','ps'+str(i)+'4','ps'+str(i)+'5'])
        surf=np.vstack((surf,temprow))
        
    return surf

def ZMeshParam(Structure):
    temp= np.array((Structure["MeshParam"].values))
    ZMesh=np.empty((0))
    
    for layer in temp:
        if type(layer)==tuple:
            ZMesh=np.concatenate((ZMesh, [str(layer[0]) +' Using Progression ' + str(layer[1])]))
            
        else:
            ZMesh=np.concatenate((ZMesh,[str(layer)] ))
        
    return ZMesh

def XYMeshParam(Structure):
    XY=np.empty((0))
    for i in range(len(Structure)+1):
        XY=np.concatenate((XY,['tf'] ))
    
    return XY

 #(*****************************************************************************************************************************************)
#(* Variables used in geo file *)
#(*****************************************************************************************************************************************)"
        
OriginalNumLayers=(len(Structure.index)+1)
OriginalThicknessLayer=np.insert(Structure["nm"].values,0,0)
Structure=MeshConstructor(Structure)
Numoflayer=len(Structure.index)
LayersType=Structure["type"].values
LayersThickness=Structure["nm"].values

#Height of each layer
Height=y = np.array([sum(LayersThickness[:i+1]) for i in range(len(LayersThickness))])
Height=np.insert(Height,0,0)


NVirt=(len(Structure.index)+2)
row= np.empty((0,4))
lines=np.empty((0,8))
lineloop=np.empty((0,5))
surf=np.empty((0,5))
        
PointStructure=Pointfunc(NVirt,row)    
LineStructure=Linefunc(NVirt,lines)  
LineLoopStructure=Lineloopfunc(NVirt,lineloop)
SurfaceStructure=Surfacefunc(NVirt,surf)

PointStructure[0]=['p1','p2','p3','p4']
LineStructure[0]=['l1','l2','l3','l4','l5','l6','l7','l8']
LineLoopStructure[0]=['ll1','ll2','ll3','ll4','ll5']        
SurfaceStructure[0]=['ps1','ps2','ps3','ps4','ps5']


ZMesh=ZMeshParam(Structure)
XYMesh=XYMeshParam(Structure)  
    
 #(*****************************************************************************************************************************************)
#(* Generate geo file *)
#(*****************************************************************************************************************************************)"#         

f=open(Name, "a")
f.write(
        "/////GeneratedbyPython\n\
/////ClaytonQwah2018\n\
\n\
la = 2*10^-6; // 10 nm\n\
u = 10^-4; // um \n\
n = 10^-7; // nm \n\
a = 10^-8; // a \n\
length = "+str(DeviceLength)+" *n; // nm \n\
mesh_x = "+str(LateralMesh)+" * n; \n\
tf = length/mesh_x; \n\
\n\
tf_qw = 15; \n\
tf_cap = 10; \n\
// // // // // // // // // // // // // // // // // // // // // // \n\
                    // // SUBSTRATE LAYER // // // / \n\
// // // // //// // // // // // // // // // // // // // // // // \n\
\n\
// // Define Points // // // / \n\
\n\
"+str(PointStructure[0][0])+" = newp; Point (" + str(PointStructure[0][0])+") = {0, 0, 0, la};\n\
"+str(PointStructure[0][1])+" = newp; Point (" + str(PointStructure[0][1])+") = {length, 0, 0, la};\n\
"+str(PointStructure[0][2])+" = newp; Point (" + str(PointStructure[0][2])+") = {length, length, 0, la};\n\
"+str(PointStructure[0][3])+" = newp; Point (" + str(PointStructure[0][3])+") = {0, length, 0, la};\n\
\n\
// // Define Line // // // /\n\
\n\
l1 = newl; Line (l1) = {p1, p2};\n\
l2 = newl; Line (l2) = {p2, p3};\n\
l3 = newl; Line (l3) = {p3, p4};\n\
l4 = newl; Line (l4) = {p4, p1};\n\
\n\
ll1 = newll; Line Loop (ll1) = {l1, l2, l3, l4};\n\
ps1 = news; Plane Surface (ps1) = {ll1};\n\
\n\
// // Define Transfinite Mesh // // // /\n\
\n\
Transfinite Line {l1} = "+str(XYMesh[0])+"+1;\n\
Transfinite Line {l2} = "+str(XYMesh[0])+"+1;\n\
Transfinite Line {l3} = "+str(XYMesh[0])+"+1;\n\
Transfinite Line {l4} = "+str(XYMesh[0])+"+1;\n\
\n\
Transfinite Surface {ps1} = {1, 2, 3, 4};")
for i in range(1,NVirt-1):
        f.write("\n\
// // // // // // // // // // // // // // // // // // // // // / \n\
                    // ** ***NEW VOL "+str(i+1)+" ***  // // // /\n\
// // // // // // // // // // // // // // // // // // // // // /\n\
\n\
// // Define Points // // // /\n\
\n\
"+str(PointStructure[i][0])+" = newp;Point("+str(PointStructure[i][0])+") = {0, 0, "+str(int(Height[i]))+".*n, la};\n\
"+str(PointStructure[i][1])+" = newp;Point("+str(PointStructure[i][1])+" ) = {length, 0,"+str(int(Height[i]))+".*n, la};\n\
"+str(PointStructure[i][2])+" = newp;Point("+str(PointStructure[i][2])+") = {length, length,"+str(int(Height[i]))+".*n, la};\n\
"+str(PointStructure[i][3])+" = newp;Point("+str(PointStructure[i][3])+") = {0, length, "+str(int(Height[i]))+".*n, la};\n\
\n\
// // Define Lines and Volume /// \n\
\n\
"+str(LineStructure[i][0])+" = newl;Line("+str(LineStructure[i][0])+")={"+str(PointStructure[i][0])+","+str(PointStructure[i][1])+"};\n\
"+str(LineStructure[i][1])+" = newl;Line("+str(LineStructure[i][1])+")={"+str(PointStructure[i][1])+","+str(PointStructure[i][2])+"};\n\
"+str(LineStructure[i][2])+" = newl;Line("+str(LineStructure[i][2])+")={"+str(PointStructure[i][2])+","+str(PointStructure[i][3])+"};\n\
"+str(LineStructure[i][3])+" = newl;Line("+str(LineStructure[i][3])+")={"+str(PointStructure[i][3])+","+str(PointStructure[i][0])+"};\n\
"+str(LineStructure[i][4])+" = newl;Line("+str(LineStructure[i][4])+")={"+str(PointStructure[i-1][0])+","+str(PointStructure[i][0])+"};\n\
"+str(LineStructure[i][5])+" = newl;Line("+str(LineStructure[i][5])+")={"+str(PointStructure[i-1][1])+","+str(PointStructure[i][1])+"};\n\
"+str(LineStructure[i][6])+" = newl;Line("+str(LineStructure[i][6])+")={"+str(PointStructure[i-1][2])+","+str(PointStructure[i][2])+"};\n\
"+str(LineStructure[i][7])+" = newl;Line("+str(LineStructure[i][7])+")={"+str(PointStructure[i-1][3])+","+str(PointStructure[i][3])+"};\n\
\n\
\n\
// // // // // // // // // // // // //\n\
 // Mesh Definition of Volume "+str(i+1)+" // \n\
// // // // // // // // // // // // //\n\
\n\
Transfinite Line {"+str(LineStructure[i][0])+"} = "+str(XYMesh[i])+"+1;\n\
Transfinite Line {"+str(LineStructure[i][1])+"} = "+str(XYMesh[i])+"+1;\n\
Transfinite Line {"+str(LineStructure[i][2])+"} = "+str(XYMesh[i])+"+1;\n\
Transfinite Line {"+str(LineStructure[i][3])+"} = "+str(XYMesh[i])+"+1;\n\
Transfinite Line {"+str(LineStructure[i][4])+"} = "+str(ZMesh[i-1])+";\n\
Transfinite Line {"+str(LineStructure[i][5])+"} = "+str(ZMesh[i-1])+";\n\
Transfinite Line {"+str(LineStructure[i][6])+"} = "+str(ZMesh[i-1])+";\n\
Transfinite Line {"+str(LineStructure[i][7])+"} = "+str(ZMesh[i-1])+";\n\
\n\
"+str(LineLoopStructure[i][0])+" = newll; Line Loop ("+str(LineLoopStructure[i][0])+") = {"+str(LineStructure[i][0])+","+str(LineStructure[i][1])+","+str(LineStructure[i][2])+","+str(LineStructure[i][3])+"};\n\
"+str(LineLoopStructure[i][1])+" = newll; Line Loop ("+str(LineLoopStructure[i][1])+") = {"+str(LineStructure[i-1][0])+","+str(LineStructure[i][5])+",-"+str(LineStructure[i][0])+",-"+str(LineStructure[i][4])+"};\n\
"+str(LineLoopStructure[i][2])+" = newll; Line Loop ("+str(LineLoopStructure[i][2])+") = {"+str(LineStructure[i-1][1])+","+str(LineStructure[i][6])+",-"+str(LineStructure[i][1])+",-"+str(LineStructure[i][5])+"};\n\
"+str(LineLoopStructure[i][3])+" = newll; Line Loop ("+str(LineLoopStructure[i][3])+") = {"+str(LineStructure[i-1][2])+","+str(LineStructure[i][7])+",-"+str(LineStructure[i][2])+",-"+str(LineStructure[i][6])+"};\n\
"+str(LineLoopStructure[i][4])+" = newll; Line Loop ("+str(LineLoopStructure[i][4])+") = {"+str(LineStructure[i-1][3])+","+str(LineStructure[i][4])+",-"+str(LineStructure[i][3])+",-"+str(LineStructure[i][7])+"};\n\
\n\
"+str(SurfaceStructure[i][0])+" = news; Plane Surface ("+str(SurfaceStructure[i][0])+") = {"+str(LineLoopStructure[i][0])+"};\n\
"+str(SurfaceStructure[i][1])+" = news; Plane Surface ("+str(SurfaceStructure[i][1])+") = {"+str(LineLoopStructure[i][1])+"};\n\
"+str(SurfaceStructure[i][2])+" = news; Plane Surface ("+str(SurfaceStructure[i][2])+") = {"+str(LineLoopStructure[i][2])+"};\n\
"+str(SurfaceStructure[i][3])+" = news; Plane Surface ("+str(SurfaceStructure[i][3])+") = {"+str(LineLoopStructure[i][3])+"};\n\
"+str(SurfaceStructure[i][4])+" = news; Plane Surface ("+str(SurfaceStructure[i][4])+") = {"+str(LineLoopStructure[i][4])+"};\n\
\n\
Transfinite Surface {"+str(SurfaceStructure[i][0])+"} ={"+str(PointStructure[i][0])+","+str(PointStructure[i][1])+","+str(PointStructure[i][2])+","+str(PointStructure[i][3])+"};\n\
Transfinite Surface {"+str(SurfaceStructure[i][1])+"} ={"+str(PointStructure[i-1][0])+","+str(PointStructure[i-1][1])+","+str(PointStructure[i][1])+","+str(PointStructure[i][0])+"};\n\
Transfinite Surface {"+str(SurfaceStructure[i][2])+"} ={"+str(PointStructure[i-1][1])+","+str(PointStructure[i-1][2])+","+str(PointStructure[i][2])+","+str(PointStructure[i][1])+"};\n\
Transfinite Surface {"+str(SurfaceStructure[i][3])+"} ={"+str(PointStructure[i-1][3])+","+str(PointStructure[i-1][2])+","+str(PointStructure[i][2])+","+str(PointStructure[i][3])+"};\n\
Transfinite Surface {"+str(SurfaceStructure[i][4])+"} ={"+str(PointStructure[i-1][0])+","+str(PointStructure[i-1][3])+","+str(PointStructure[i][3])+","+str(PointStructure[i][0])+"};\n\
\n\
Surface Loop ("+str(i)+") ={"+str(SurfaceStructure[i-1][0])+","+str(SurfaceStructure[i][0])+","+str(SurfaceStructure[i][1])+","+str(SurfaceStructure[i][2])+","+str(SurfaceStructure[i][3])+","+str(SurfaceStructure[i][4])+"};\n\
Volume ("+str(i)+") = {"+str(i)+"};\n\
Transfinite Volume ("+str(i)+");\n\
")
f.close()


   
