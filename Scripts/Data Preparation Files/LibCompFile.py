# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:13:24 2020

@author: Clayton
"""

import shutil
import os
import fileinput

target_directory='C:\\Users\\Clayton\\Documents\\GitHub\\3D_DDCC_Analysis\\Scripts\\Data Preparation Files\\'
Name='libcompositionmapgen.f90'
original = 'D:\\CNSI\\Basis\\libcompositionmapgen.f90'
target = target_directory+Name
shutil.copyfile(original, target)


Lx=300.0 
Ly=300.0 
Lqw=30.0 
Lba=80.0 
Lebl=100.0
qwnum=1
capnum=1
eblnum=1
Incomp_qw=0.22
Alcomp_ebl=0.00
Alcomp_cap=0.20
L_nGaN=400.0
Lcap=20.0
largeW=18 
sigma=2.0
gwindow=3.0

os.chdir(target_directory)
#if os.path.isfile(Name):
#    print ("File exist")
#    os.remove(Name)
#else:
#    print ("File not exist")
f=open(Name, "a")
f.close()

 