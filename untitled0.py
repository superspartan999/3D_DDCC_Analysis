# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:24:14 2018

@author: Kun
"""

import numpy as np
import time as time
import math as math
from math import exp
from math import log
import random as rd
import scipy as sp
import numpy as np


from rates import *
from processes import *
from miscellaneous import *
from properties import *
import time
import csv
from itertools import izip

from numpy import genfromtxt


spec1data = genfromtxt('spec1coords.csv', delimiter=',')
spec2data = genfromtxt('spec2coords.csv', delimiter=',')
wiretopdata=genfromtxt('wiretop.csv',delimiter=',')
samedata=genfromtxt('samespeciesneighbours.csv', delimiter=',')
diffdata=genfromtxt('diffspeciesneighbours.csv', delimiter=',')
sametop=genfromtxt('sameneighbourtop.csv', delimiter=',')
difftop=genfromtxt('diffneighbourtop.csv', delimiter=',')


xposition = 5
yposition = 5
dropletx = 30
droplety = 30

surfacex = 50
surfacey = 50
surfacez = 1000

species1bools=np.zeros((surfacex,surfacey,surfacez), dtype=bool)
species2bools=np.zeros((surfacex,surfacey,surfacez), dtype=bool)
surfaceint= np.zeros((surfacex,surfacey,surfacez), dtype=bool)
samespeciesneighbours=np.zeros((surfacex,surfacey,surfacez), dtype=int)
diffspeciesneighbours=np.zeros((surfacex,surfacey,surfacez), dtype=int)

wiretop=np.empty((wiretopdata.shape[0],wiretopdata.shape[1]), dtype= int)
sameneighbourtop=np.empty((wiretop.shape[0],wiretop.shape[1]), dtype= int)
diffneighbourtop=np.empty((wiretop.shape[0],wiretop.shape[1]), dtype= int)

for k in range(spec1data.shape[0]):

    xcoord = spec1data[k][0]
    ycoord = spec1data[k][1]
    zcoord = spec1data[k][2]

    species1bools[xcoord][ycoord][zcoord]=1

for k in range(spec2data.shape[0]):
    xcoord = spec2data[k][0]
    ycoord = spec2data[k][1]
    zcoord = spec2data[k][2]
    species2bools[xcoord][ycoord][zcoord]=1

for k in range(samedata.shape[0]):

    xcoord = samedata[k][0]
    ycoord = samedata[k][1]
    zcoord = samedata[k][2]
    samespeciesneighbours[xcoord][ycoord][zcoord]=samedata[k][3]

for k in range(diffdata.shape[0]):
    xcoord = diffdata[k][0]
    ycoord = diffdata[k][1]
    zcoord = diffdata[k][2]
    diffspeciesneighbours[xcoord][ycoord][zcoord]=diffdata[k][3]

for i in range(wiretopdata.shape[0]):
    for j in range(wiretopdata.shape[1]):
        wiretop[i][j]= wiretopdata[i][j]


for i in range(sametop.shape[0]):
    for j in range(sametop.shape[1]):
        sameneighbourtop[i][j]= sametop[i][j]


for i in range(wiretopdata.shape[0]):
    for j in range(wiretopdata.shape[1]):
        diffneighbourtop[i][j]= difftop[i][j]


start_time = time.time()

tracker = 0
counter = 0
processarray=np.zeros((98,2), dtype=int)

for i in range(1,98):
    processarray[i][0]=i


    #plot1scatteronceattheend(surfacewire)
while counter <atomnumber:
    moveornot=0

    #print "counter", counter
    #s1=np.sum(surfaceint)
    #s2=np.sum(surfacewire)
    filledelements=np.count_nonzero(wiretop)

    r1 = Rjump1s(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r2 = Rjump1d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r3 = Rjump1s1d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r4 = Rjump1s2d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r5 = Rjump1s3d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r6 = Rjump1s4d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r7 = Rjump2s1d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r8 = Rjump2s2d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r9 = Rjump2s3d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r10 = Rjump3s1d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r11 = Rjump3s2d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r12 = Rjump4s1d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r13 = Rjump5s(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r14 = Rjump5d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r15 = Rjump2d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r16 = Rjump2s(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r17 = Rjump3d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r18 = Rjump3s(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r19 = Rjump4d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)
    r20 = Rjump4s(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species1bools,wiretop,species = 1)


    r21 = Rexchange1s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r22 = Rexchange1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r23 = Rexchange1s1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r24 = Rexchange1s2d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r25 = Rexchange1s3d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r26 = Rexchange1s4d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r27 = Rexchange2s1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r28 = Rexchange2s2d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r29 = Rexchange2s3d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r30 = Rexchange3s1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r31 = Rexchange3s2d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r32 = Rexchange4s1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r33 = Rexchange5s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r34 = Rexchange5d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r35 = Rexchange2d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r36 = Rexchange2s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r37 = Rexchange3d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r38 = Rexchange3s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r39 = Rexchange4d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r40 = Rexchange4s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r41 = Rexchange6s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r42 = Rexchange6d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r43 = Rexchange1s5d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r44 = Rexchange5s1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r45 = Rexchange4s2d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r46 = Rexchange2s4d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)
    r47 = Rexchange3s3d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species1bools,species=1)

    r48 = Rjump1s(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r49 = Rjump1d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r50 = Rjump1s1d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r51 = Rjump1s2d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r52 = Rjump1s3d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r53 = Rjump1s4d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r54 = Rjump2s1d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r55 = Rjump2s2d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r56 = Rjump2s3d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r57 = Rjump3s1d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r58 = Rjump3s2d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r59 = Rjump4s1d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r60 = Rjump5s(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r61 = Rjump5d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r62 = Rjump2d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r63 = Rjump2s(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r64 = Rjump3d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r65 = Rjump3s(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r66 = Rjump4d(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)
    r67 = Rjump4s(dropletx,droplety,xposition,yposition,sameneighbourtop,diffneighbourtop,species2bools,wiretop,species = 2)

    r68 = Rexchange1s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r69 = Rexchange1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,2)
    r70 = Rexchange1s1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r71 = Rexchange1s2d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r72 = Rexchange1s3d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r73 = Rexchange1s4d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r74 = Rexchange2s1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r75 = Rexchange2s2d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r76 = Rexchange2s3d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r77 = Rexchange3s1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r78 = Rexchange3s2d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r79 = Rexchange4s1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r80 = Rexchange5s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r81 = Rexchange5d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r82 = Rexchange2d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r83 = Rexchange2s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r84 = Rexchange3d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r85 = Rexchange3s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r86 = Rexchange4d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r87 = Rexchange4s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r88 = Rexchange6s(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r89 = Rexchange6d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r90 = Rexchange1s5d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r91 = Rexchange5s1d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r92 = Rexchange4s2d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r93 = Rexchange2s4d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)
    r94 = Rexchange3s3d(dropletx,droplety,xposition,yposition,wiretop,samespeciesneighbours,diffspeciesneighbours,species2bools,species=2)

    rsum15 = r1 + r2 + r3 + r4 + r5
    rsum510 = r6 + r7 + r8 + r9 + r10
    rsum1015 = r11 + r12 + r13 + r14 + r15
    rsum1520 = r16 + r17 + r18 + r19 + r20

    Rsum2125=r21+r22+r23+r24+r25
    Rsum2630=r26+r27+r28+r29+r30
    Rsum3135=r31+r32+r33+r34+r35
    Rsum3640=r36+r37+r38+r39+r40
    Rsum4145=r41+r42+r43+r44+r45
    Rsum4650=r46+r47+r48+r49+r50
    Rsum5155=r51+r52+r53+r54+r55
    Rsum5660=r56+r57+r58+r59+r60
    Rsum6165=r61+r62+r63+r64+r65
    Rsum6670=r66+r67+r68+r69+r70
    Rsum7175=r71+r72+r73+r74+r75
    Rsum7680=r76+r77+r78+r79+r80
    Rsum8185=r81+r82+r83+r84+r85
    Rsum8690=r86+r87+r88+r89+r90

    Rsum2145=Rsum2125+Rsum2630+Rsum3135+Rsum3640+Rsum4145



    Radd=Rdep1(dropletx,droplety)+Rdep2(dropletx,droplety)
    Rjump=Radd+rsum15+rsum510+rsum1015+rsum1520

    partialsums[0]=Rdep1(dropletx,droplety)
    partialsums[1]=Radd
    partialsums[2]=Radd+ r1
    partialsums[3]=Radd+ r1 + r2
    partialsums[4]=Radd+ r1 + r2 + r3
    partialsums[5]=Radd+ r1 + r2 + r3 + r4
    partialsums[6]=Radd+ r1 + r2 + r3 + r4 + r5
    partialsums[7]=Radd+ rsum15 + r6
    partialsums[8]=Radd+ rsum15 + r6 + r7
    partialsums[9]=Radd+ rsum15 + r6 + r7 + r8
    partialsums[10]=Radd+ rsum15 + r6 + r7 + r8 + r9
    partialsums[11]=Radd+ rsum15 + r6 + r7 + r8 + r9 + r10
    partialsums[12]=Radd+ rsum15 + rsum510 + r11
    partialsums[13]=Radd+ rsum15 + rsum510 + r11 + r12
    partialsums[14]=Radd+ rsum15 + rsum510 + r11 + r12 + r13
    partialsums[15]=Radd+ rsum15 + rsum510 + r11 + r12 + r13 + r14
    partialsums[16]=Radd+ rsum15 + rsum510 + r11 + r12 + r13 + r14 + r15
    partialsums[17]=Radd+ rsum15 + rsum510 + rsum1015 + r16
    partialsums[18]=Radd+ rsum15 + rsum510 + rsum1015 + r16 + r17
    partialsums[19]=Radd+ rsum15 + rsum510 + rsum1015 + r16 + r17 + r18
    partialsums[20]=Radd+ rsum15 + rsum510 + rsum1015 + r16 + r17 + r18 + r19
    partialsums[21]=Radd+ rsum15 + rsum510 + rsum1015 + rsum1520

    partialsums[22]=Rjump+r21
    partialsums[23]=Rjump+r21+r22
    partialsums[24]=Rjump+r21+r22+r23
    partialsums[25]=Rjump+r21+r22+r23+r24
    partialsums[26]=Rjump+r21+r22+r23+r24+r25
    partialsums[27]=Rjump+Rsum2125+r26
    partialsums[28]=Rjump+Rsum2125+r26+r27
    partialsums[29]=Rjump+Rsum2125+r26+r27+r28
    partialsums[30]=Rjump+Rsum2125+r26+r27+r28+r29
    partialsums[31]=Rjump+Rsum2125+r26+r27+r28+r29+r30
    partialsums[32]=Rjump+Rsum2125+Rsum2630+r31
    partialsums[33]=Rjump+Rsum2125+Rsum2630+r31+r32
    partialsums[34]=Rjump+Rsum2125+Rsum2630+r31+r32+r33
    partialsums[35]=Rjump+Rsum2125+Rsum2630+r31+r32+r33+r34
    partialsums[36]=Rjump+Rsum2125+Rsum2630+r31+r32+r33+r34+r35
    partialsums[37]=Rjump+Rsum2125+Rsum2630+Rsum3135+r36
    partialsums[38]=Rjump+Rsum2125+Rsum2630+Rsum3135+r36+r37
    partialsums[39]=Rjump+Rsum2125+Rsum2630+Rsum3135+r36+r37+r38
    partialsums[40]=Rjump+Rsum2125+Rsum2630+Rsum3135+r36+r37+r38+r39
    partialsums[41]=Rjump+Rsum2125+Rsum2630+Rsum3135+r36+r37+r38+r39+r40
    partialsums[42]=Rjump+Rsum2125+Rsum2630+Rsum3135+Rsum3640+r41
    partialsums[43]=Rjump+Rsum2125+Rsum2630+Rsum3135+Rsum3640+r41+r42
    partialsums[44]=Rjump+Rsum2125+Rsum2630+Rsum3135+Rsum3640+r41+r42+r43
    partialsums[45]=Rjump+Rsum2125+Rsum2630+Rsum3135+Rsum3640+r41+r42+r43+r44
    partialsums[46]=Rjump+Rsum2125+Rsum2630+Rsum3135+Rsum3640+r41+r42+r43+r44+r45
    partialsums[47]=Rjump+Rsum2125+Rsum2630+Rsum3135+Rsum3640+Rsum4145+r46

    partialsums[48]=Rjump+Rsum2145+r46+r47
    partialsums[49]=Rjump+Rsum2145+r46+r47+r48
    partialsums[50]=Rjump+Rsum2145+r46+r47+r48+r49
    partialsums[51]=Rjump+Rsum2145+r46+r47+r48+r49+r50
    partialsums[52]=Rjump+Rsum2145+Rsum4650 + r51
    partialsums[53]=Rjump+Rsum2145+Rsum4650 + r51 + r52
    partialsums[54]=Rjump+Rsum2145+Rsum4650 + r51 + r52 + r53
    partialsums[55]=Rjump+Rsum2145+Rsum4650 + r51 + r52 + r53 + r54
    partialsums[56]=Rjump+Rsum2145+Rsum4650 + r51 + r52 + r53 + r54 + r55
    partialsums[57]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + r56
    partialsums[58]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + r56 + r57
    partialsums[59]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + r56 + r57 + r58
    partialsums[60]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + r56 + r57 + r58 + r59
    partialsums[61]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + r56 + r57 + r58 + r59 + r60
    partialsums[62]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + r61
    partialsums[63]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + r61 + r62
    partialsums[64]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + r61 + r62 + r63
    partialsums[65]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + r61 + r62 + r63 + r64
    partialsums[66]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + r61 + r62 + r63 + r64 + r65
    partialsums[67]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + r66
    partialsums[68]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + r66 + r67
    partialsums[69]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + r66 + r67 + r68
    partialsums[70]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + r66 + r67 + r68 + r69

    partialsums[71]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + r66 + r67 + r68 + r69 + r70
    partialsums[72]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + r71
    partialsums[73]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + r71 + r72
    partialsums[74]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + r71 + r72 + r73
    partialsums[75]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + r71 + r72 + r73 + r74
    partialsums[76]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + r71 + r72 + r73 + r74 + r75
    partialsums[77]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + r76
    partialsums[78]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + r76 + r77
    partialsums[79]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + r76 + r77 + r78
    partialsums[80]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + r76 + r77 + r78 + r79
    partialsums[81]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + r76 + r77 + r78 + r79 + r80
    partialsums[82]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + r81
    partialsums[83]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + r81 + r82
    partialsums[84]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + r81 + r82 + r83
    partialsums[85]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + r81 + r82 + r83 + r84
    partialsums[86]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + r81 + r82 + r83 + r84 + r85
    partialsums[87]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + Rsum8185 + r86
    partialsums[88]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + Rsum8185 + r86 + r87
    partialsums[89]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + Rsum8185 + r86 + r87 + r88
    partialsums[90]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + Rsum8185 + r86 + r87 + r88 + r89
    partialsums[91]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + Rsum8185 + r86 + r87 + r88 + r89 + r90
    partialsums[92]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + Rsum8185 + Rsum8690 + r91
    partialsums[93]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + Rsum8185 + Rsum8690 + r91 + r92
    partialsums[94]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + Rsum8185 + Rsum8690 + r91 + r92+r93
    partialsums[95]=Rjump+Rsum2145+Rsum4650 + Rsum5155 + Rsum5660 + Rsum6165 + Rsum6670 + Rsum7175 + Rsum7680 + Rsum8185 + Rsum8690 + r91 + r92+r93+r94


    #partialsums[4]=Radd()+Rmove(s1,s2)+Rdep(surfacevolume,dropletvolume,s1,s2)+Rjump(filledelements)+Rswitch(s2)
    Rtot=Radd+ rsum15 + rsum510 + rsum1015 + rsum1520+ Rsum2125+Rsum2630+Rsum3135+Rsum3640+Rsum4145+Rsum4650+Rsum5155+Rsum5660+Rsum6165+Rsum6670+Rsum7175+Rsum7680+Rsum8185+Rsum8690+r91+r92+r93+r94


    randomnumber=rd.random()
    randomR=randomnumber*Rtot
    #print randomR, Rtot, partialsums[1]
    process=0


    for k in range(0,len(partialsums)):
        # if k==0:
        #  print partialsums[k], randomR
        # if k>0:
        #  print partialsums[k], randomR, k, partialsums[k]-partialsums[k-1]
        if partialsums[k] >  randomR :
            process=k+1


            break




    #print process, 'process'

    if process ==0:
        break


    if process == 1:
        processarray[1][1]+=1
        x=rd.randrange(xposition, xposition+dropletx, 1)
        y=rd.randrange(yposition, yposition+droplety, 1)

        #p=wiretop[x][y]
        #print wiretop[x][y]
        surfacewire[x][y][wiretop[x][y]]=1
        surfaceint[x][y][wiretop[x][y]]=1
        species2bools[x][y][wiretop[x][y]] = 1
        species2wire[x][y][wiretop[x][y]] = 1





        updated= updatepoint(x,y,wiretop[x][y],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop, wiretop)

        samespeciesneighbours=updated[0][:]
        diffspeciesneighbours=updated[1][:]
        sameneighbourtop=updated[2][:]
        diffneighbourtop=updated[3][:]

        #updatearound=updateneighbours(x,y,wiretop[x][y],surfacewire,species1bools,species2bools,samespeciesneighbours,diffspeciesneighbours,sameneighbourtop,diffneighbourtop,wiretop)
        #samespeciesneighbours=updatearound[0][:]
        #diffspeciesneighbours=updatearound[1][:]


        wiretop[x][y]=wiretop[x][y]+1



#        if wiretop[x][y]==0:# and species1bools[x][y][0]==0:
#
#            surfacewire[x][y][0]=1
#            surfaceint[x][y][0]=1
#            species2bools[x][y][0] = 1
#            species2wire[x][y][0] = 1
#            wiretop[x][y]=1
#
#
#
#            updated= updatepoint(x,y,0,surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop, wiretop)
#
#            samespeciesneighbours=updated[0][:]
#            diffspeciesneighbours=updated[1][:]
#            sameneighbourtop=updated[2][:]
#            diffneighbourtop=updated[3][:]
#
#            #updatearound=updateneighbours(x,y,0,surfacewire,species1bools,species2bools,samespeciesneighbours,diffspeciesneighbours,sameneighbourtop,diffneighbourtop,wiretop)
#            #samespeciesneighbours=updatearound[0][:]
#            #diffspeciesneighbours=updatearound[1][:]
#
#
#            #print wiretop[x][y]
#
#
#
#        elif wiretop[x][y] !=0 and species1bools[x][y][wiretop[x][y]]==0:
#            #p=wiretop[x][y]
#            #print wiretop[x][y]
#            surfacewire[x][y][wiretop[x][y]]=1
#            surfaceint[x][y][wiretop[x][y]]=1
#            species2bools[x][y][wiretop[x][y]] = 1
#            species2wire[x][y][wiretop[x][y]] = 1
#
#
#
#
#
#            updated= updatepoint(x,y,wiretop[x][y],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop, wiretop)
#
#            samespeciesneighbours=updated[0][:]
#            diffspeciesneighbours=updated[1][:]
#            sameneighbourtop=updated[2][:]
#            diffneighbourtop=updated[3][:]
#
#            #updatearound=updateneighbours(x,y,wiretop[x][y],surfacewire,species1bools,species2bools,samespeciesneighbours,diffspeciesneighbours,sameneighbourtop,diffneighbourtop,wiretop)
#            #samespeciesneighbours=updatearound[0][:]
#            #diffspeciesneighbours=updatearound[1][:]
#
#
#            wiretop[x][y]=wiretop[x][y]+1



    if process == 2:
        processarray[2][1]+=1
        x=rd.randrange(xposition, xposition+dropletx, 1)
        y=rd.randrange(yposition, yposition+droplety, 1)

        #p=wiretop[x][y]
        #print wiretop[x][y]
        surfacewire[x][y][wiretop[x][y]]=1
        surfaceint[x][y][wiretop[x][y]]=1
        species1bools[x][y][wiretop[x][y]] = 1
        species1wire[x][y][wiretop[x][y]] = 1





        updated= updatepoint(x,y,wiretop[x][y],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop, wiretop)

        samespeciesneighbours=updated[0][:]
        diffspeciesneighbours=updated[1][:]
        sameneighbourtop=updated[2][:]
        diffneighbourtop=updated[3][:]

        #updatearound=updateneighbours(x,y,wiretop[x][y],surfacewire,species1bools,species2bools,samespeciesneighbours,diffspeciesneighbours,sameneighbourtop,diffneighbourtop,wiretop)
        #samespeciesneighbours=updatearound[0][:]
        #diffspeciesneighbours=updatearound[1][:]


        wiretop[x][y]=wiretop[x][y]+1

#
#        if wiretop[x][y]==0 and species2bools[x][y][0]==0:
#
#            surfacewire[x][y][0]=1
#            surfaceint[x][y][0]=1
#            species1bools[x][y][0] = 1
#            species1wire[x][y][0] = 1
#            wiretop[x][y]=1
#
#
#            updated= updatepoint(x,y,wiretop[x][y],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop, wiretop)
#
#            samespeciesneighbours=updated[0][:]
#            diffspeciesneighbours=updated[1][:]
#            sameneighbourtop=updated[2][:]
#            diffneighbourtop=updated[3][:]
##
##            updatearound=updateneighbours(x,y,0,surfacewire,species1bools,species2bools,samespeciesneighbours,diffspeciesneighbours,sameneighbourtop,diffneighbourtop,wiretop)
##            samespeciesneighbours=updatearound[0][:]
##            diffspeciesneighbours=updatearound[1][:]
#
#
#
#            print wiretop[x][y]
#
#
#
#        elif wiretop[x][y] !=0 and species2bools[x][y][wiretop[x][y]]==0:
#            #p=wiretop[x][y]
#            #print wiretop[x][y]
#            surfacewire[x][y][wiretop[x][y]]=1
#            surfaceint[x][y][wiretop[x][y]]=1
#            species1bools[x][y][wiretop[x][y]] = 1
#            species1wire[x][y][wiretop[x][y]] = 1
#
#
#            updated= updatepoint(x,y,wiretop[x][y],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop, wiretop)
#
#            samespeciesneighbours=updated[0][:]
#            diffspeciesneighbours=updated[1][:]
#            sameneighbourtop=updated[2][:]
#            diffneighbourtop=updated[3][:]
##
##            updatearound=updateneighbours(x,y,wiretop[x][y],surfacewire,species1bools,species2bools,samespeciesneighbours,diffspeciesneighbours,sameneighbourtop,diffneighbourtop,wiretop)
##            samespeciesneighbours=updatearound[0][:]
##            diffspeciesneighbours=updatearound[1][:]
#
#            wiretop[x][y]=wiretop[x][y]+1

#-------------------------------------------------------------------------------------------------------------------------------------------------------

    if counter%100== 0:
        surface3dbools = updatedroplet(dropletx, droplety, dropletz,surfacex,surfacey,surfacez, xposition, yposition,zposition, wiretop)
#--------------------------------------------------------------------------------------------------------------------------------------------------------
    if 3 <= process<= 22:
        neigh=nsamendiff(process)
        #print neigh[0], neigh[1], 'in main'
        processarray[process][1]+=1
        pick = pickjumpatom(xposition,yposition,dropletx,droplety,wiretop,sameneighbourtop,diffneighbourtop,neigh[0],neigh[1],species1bools)
        jumpdestination = findnewpos(pick[0],pick[1],pick[2],xposition, yposition, dropletx, droplety, wiretop,surfacewire)

        surfacewire = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],surfacewire)
        surfaceint = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],surfaceint)

        species1bools = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species1bools)
        species1wire = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species1wire)
#

#        if checkspecies(pick[0],pick[1],pick[2],species1bools,species2bools) == 1:
#
#            species1bools = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species1bools)
#            species1wire = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species1wire)
#
#        elif checkspecies(pick[0],pick[1],pick[2],species1bools,species2bools) == 2:
            #species2wire = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species2wire)
            #species2bools = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species2bools)

        updatedold= updatedeposit(pick[0],pick[1],pick[2],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop)
        samespeciesneighbours=updatedold[0][:]
        diffspeciesneighbours=updatedold[1][:]
        sameneighbourtop=updatedold[2][:]
        diffneighbourtop=updatedold[3][:]

        updatednew= updatedeposit(jumpdestination[0],jumpdestination[1],jumpdestination[2],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop)
        samespeciesneighbours=updatednew[0][:]
        diffspeciesneighbours=updatednew[1][:]
        sameneighbourtop=updatednew[2][:]
        diffneighbourtop=updatednew[3][:]
#-------------------------------------------------------------------------------------------------------------------------------------------------------
    if 23<=process<=49 :
        neigh=nexchangesamediff(process)
        #print neigh[0], neigh[1], 'in main'
        processarray[process][1]+=1

        pick = pickexchangeatom(xposition,yposition,dropletx,droplety,wiretop,samespeciesneighbours,diffspeciesneighbours,neigh[0],neigh[1],species1bools)


        exchangedestination = findexchangepos(pick[0],pick[1],pick[2],xposition, yposition, dropletx, droplety, wiretop,species2bools)

        # expos=0
        # while expos==0:
        #if species1bools[pick[0]][pick[1]][pick[2]]==1:
        #    exchangedestination = findexchangepos(pick[0],pick[1],pick[2],xposition, yposition, dropletx, droplety, wiretop,species2bools)
        #elif species2bools[pick[0]][pick[1]][pick[2]]==1:
        #    exchangedestination = findexchangepos(pick[0],pick[1],pick[2],xposition, yposition, dropletx, droplety, wiretop,species1bools)

                # if surfacewire[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]]==1:
                #     expos=1



        #
        # try:
        #     pick = pickrandparticle(dropletx,droplety,xposition,yposition,wiretop)
        #     pickconfig = findneighboursinit(pick[0],pick[1], pick[2] ,species1bools,species2bools)

        #     exchangeconfig = findneighboursinit(exchangedestination[0],exchangedestination[1], exchangedestination[2] ,species1bools,species2bools)
        # except:
        #     continue

        #surfacewire = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],surfacewire)
        #surfaceint = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],surfaceint)

        rcheck = Arrhenius(neigh[0],neigh[1],esamespecies1 ,esamespecies2,ediffspecies1,ediffspecies2,temp,1)
        if rd.random()<rcheck:
             species1bools[pick[0]][pick[1]][pick[2]] = 0
             species1bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 1
             species2bools[pick[0]][pick[1]][pick[2]] = 1
             species2bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 0



#
#            if checkspecies(pick[0],pick[1],pick[2],species1bools,species2bools) == 1:
#
#
#                if checkspecies(exchangedestination[0],exchangedestination[1],exchangedestination[2],species1bools,species2bools) == 2:
#                    species1bools[pick[0]][pick[1]][pick[2]] = 0
#                    species1bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 1
#                    species2bools[pick[0]][pick[1]][pick[2]] = 1
#                    species2bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 0
#
#
#
#            elif checkspecies(pick[0],pick[1],pick[2],species1bools,species2bools) == 2:
#                 if checkspecies(exchangedestination[0],exchangedestination[1],exchangedestination[2],species1bools,species2bools) == 1:
                    #species1bools[pick[0]][pick[1]][pick[2]] = 1
                    #species1bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 0
                    #species2bools[pick[0]][pick[1]][pick[2]] = 0
                    #species2bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 1

        updatedold= updatepoint(pick[0],pick[1],pick[2],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop,wiretop)
        samespeciesneighbours=updatedold[0][:]
        diffspeciesneighbours=updatedold[1][:]

        updatednew= updatepoint(exchangedestination[0],exchangedestination[1],exchangedestination[2],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop,wiretop)
        samespeciesneighbours=updatednew[0][:]
        diffspeciesneighbours=updatednew[1][:]
#=----------------------------------------------------------------------------------------------------------------------------------------
    if 50<= process<= 69:
        neigh=nsamendiff(process)
        #print neigh[0], neigh[1], 'in main'
        processarray[process][1]+=1
        pick = pickjumpatom(xposition,yposition,dropletx,droplety,wiretop,sameneighbourtop,diffneighbourtop,neigh[0],neigh[1],species2bools)
        jumpdestination = findnewpos(pick[0],pick[1],pick[2],xposition, yposition, dropletx, droplety, wiretop,surfacewire)

        surfacewire = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],surfacewire)
        surfaceint = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],surfaceint)

        species2wire = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species2wire)
        species2bools = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species2bools)


#        if checkspecies(pick[0],pick[1],pick[2],species1bools,species2bools) == 1:
#
#            species1bools = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species1bools)
#            species1wire = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species1wire)
#
#        elif checkspecies(pick[0],pick[1],pick[2],species1bools,species2bools) == 2:
#            species2wire = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species2wire)
#            species2bools = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],species2bools)

        updatedold= updatedeposit(pick[0],pick[1],pick[2],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop)
        samespeciesneighbours=updatedold[0][:]
        diffspeciesneighbours=updatedold[1][:]
        sameneighbourtop=updatedold[2][:]
        diffneighbourtop=updatedold[3][:]

        updatednew= updatedeposit(jumpdestination[0],jumpdestination[1],jumpdestination[2],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop)
        samespeciesneighbours=updatednew[0][:]
        diffspeciesneighbours=updatednew[1][:]
        sameneighbourtop=updatednew[2][:]
        diffneighbourtop=updatednew[3][:]
#-------------------------------------------------------------------------------------------------------------------------------------------------------
    if 70<=process<=98 :
        neigh=nexchangesamediff(process)
        #print neigh[0], neigh[1], 'in main'
        processarray[process][1]+=1


        pick = pickexchangeatom(xposition,yposition,dropletx,droplety,wiretop,samespeciesneighbours,diffspeciesneighbours,neigh[0],neigh[1],species2bools)


        exchangedestination = findexchangepos(pick[0],pick[1],pick[2],xposition, yposition, dropletx, droplety, wiretop,species1bools)

        # expos=0
        # while expos==0:
        #if species1bools[pick[0]][pick[1]][pick[2]]==1:
        #    exchangedestination = findexchangepos(pick[0],pick[1],pick[2],xposition, yposition, dropletx, droplety, wiretop,species2bools)
        #elif species2bools[pick[0]][pick[1]][pick[2]]==1:
        #    exchangedestination = findexchangepos(pick[0],pick[1],pick[2],xposition, yposition, dropletx, droplety, wiretop,species1bools)

                # if surfacewire[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]]==1:
                #     expos=1



        #
        # try:
        #     pick = pickrandparticle(dropletx,droplety,xposition,yposition,wiretop)
        #     pickconfig = findneighboursinit(pick[0],pick[1], pick[2] ,species1bools,species2bools)

        #     exchangeconfig = findneighboursinit(exchangedestination[0],exchangedestination[1], exchangedestination[2] ,species1bools,species2bools)
        # except:
        #     continue

        #surfacewire = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],surfacewire)
        #surfaceint = jumpatom(pick[0],pick[1],pick[2],jumpdestination[0],jumpdestination[1],jumpdestination[2],surfaceint)

        rcheck = Arrhenius(neigh[0],neigh[1],esamespecies1,esamespecies2,ediffspecies1,ediffspecies2,temp,2)
        if rd.random()<rcheck:
            species1bools[pick[0]][pick[1]][pick[2]] = 1
            species1bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 0
            species2bools[pick[0]][pick[1]][pick[2]] = 0
            species2bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 1



#
#            if checkspecies(pick[0],pick[1],pick[2],species1bools,species2bools) == 1:
#
#
#                if checkspecies(exchangedestination[0],exchangedestination[1],exchangedestination[2],species1bools,species2bools) == 2:
#                    species1bools[pick[0]][pick[1]][pick[2]] = 0
#                    species1bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 1
#                    species2bools[pick[0]][pick[1]][pick[2]] = 1
#                    species2bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 0
#
#
#
#            elif checkspecies(pick[0],pick[1],pick[2],species1bools,species2bools) == 2:
#                 if checkspecies(exchangedestination[0],exchangedestination[1],exchangedestination[2],species1bools,species2bools) == 1:
#                    species1bools[pick[0]][pick[1]][pick[2]] = 1
#                    species1bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 0
#                    species2bools[pick[0]][pick[1]][pick[2]] = 0
#                    species2bools[exchangedestination[0]][exchangedestination[1]][exchangedestination[2]] = 1

        updatedold= updatepoint(pick[0],pick[1],pick[2],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop,wiretop)
        samespeciesneighbours=updatedold[0][:]
        diffspeciesneighbours=updatedold[1][:]

        updatednew= updatepoint(exchangedestination[0],exchangedestination[1],exchangedestination[2],surfacewire,species1bools,species2bools, samespeciesneighbours, diffspeciesneighbours, sameneighbourtop, diffneighbourtop,wiretop)
        samespeciesneighbours=updatednew[0][:]
        diffspeciesneighbours=updatednew[1][:]
   # if process==5:
    #         #Choose random site in the gold droplet
    #         x=rd.randrange(xposition, xposition+dropletx, 1)
    #         y=rd.randrange(yposition, yposition+droplety, 1)
    #         z=rd.randrange(zposition, zposition+wiretop[xposition][yposition]+1, 1)
    #
    #         neighbours = findneighboursinit(x,y,z,species1wire,species2wire)
    #         same = neighbours[0]
    #         diff = neighbours[1]
    #
    #         newmat= switchatom(x,y,z,surfaceint,surfacewire,moveornot,species1wire,species2wire,wiretop, time)
    #
    #         species1wire=newmat[0][:]
    #         species2wire=newmat[1][:]
    #
    #         old=findneighboursinit(x,y,z, species1bools,species2bools)
    #         new=findneighboursinit(newmat[2],newmat[3],newmat[4],species1bools, species2bools)
    #
    #         samespeciesneighbours[pick[0]][pick[1]][pick[2]]= old[0]
    #         diffspeciesneighbours[pick[0]][pick[1]][pick[2]]= old[1]
    #         samespeciesneighbours[jumpdestination[0]][jumpdestination[1]][jumpdestination[2]]= new[0]
    #         diffspeciesneighbours[jumpdestination[0]][jumpdestination[1]][jumpdestination[2]]= new[1]
    #
    #         # if diff> same:
    #         #   counter = counter + 10**(diff-same)



    h=rd.random()
    #print h, 'h'
    #print Rtot, 'rtot'
    #print log(1/h)/float(Rtot)
    #print "counter1", counter
    counter+=log(1/float(h))/float(Rtot)
    # print processarray
    print "counter2", counter
    # print time.time()-start_time, 'seconds'
    # print (time.time()-start_time)/float(60), 'minutes'
    # print (time.time()-start_time)/float(3600), 'hours'





#---------------------------------------------------------------------------------------------------------------------------------------------------------

#np.set_printoptions(threshold='nan')
end_time=time.time()
species1top=np.zeros((surfacex,surfacey),dtype=int)
species2top=np.zeros((surfacex,surfacey),dtype=int)
species1percentage=np.zeros((surfacex,surfacey),dtype=float)
species2percentage=np.zeros((surfacex,surfacey),dtype=float)
speciesdiff=np.zeros((surfacex,surfacey),dtype=int)
speciesnewsame=np.zeros((surfacex,surfacey,surfacez),dtype=int)
speciesnewdiff=np.zeros((surfacex,surfacey,surfacez),dtype=int)
specieswiretopdata=np.zeros((surfacex,surfacey,surfacez),dtype=int)

for i in range(xposition,xposition+dropletx+1):
    for j in range(yposition, yposition+droplety+1):
        for k in range(wiretop[i][j]+1):
                sameneighbours = 0
                diffneighbours = 0
                d=findpointneighbours(i,j,k,species1bools,species2bools)
                speciesnewsame[i][j][k]=d[0]
                speciesnewdiff[i][j][k]=d[1]


for i in range(xposition,xposition+dropletx):
    for j in range (yposition,yposition+droplety):
         column=0
         for k in range (surfacez):

            if species1wire[i][j][k]==1:
                species1top[i][j]+=1



            if species2wire[i][j][k]==1:
                species2top[i][j]+=1

            column+=1

         species1percentage[i][j]=(float(species1top[i][j])/float(column))*100
         species2percentage[i][j]=(float(species2top[i][j])/float(column))*100






         speciesdiff[i][j]=species1top[i][j]- species2top[i][j]


#Plots Histogram of the wire
#plothistonceattheend(surfacex,surfacey,species1top)
#print processcounter
# contourplot(surfacex,surfacey,surfacez,species2top, 'mycontour1')
# contourplot(surfacex,surfacey,surfacez,species1top, 'mycontour2')
# contourplot(surfacex,surfacey,surfacez,species2top, 'mycontour1')
# contourplot(surfacex,surfacey,surfacez,species1percentage, 'mycontour1p')
# contourplot(surfacex,surfacey,surfacez,species2percentage, 'mycontour2p')
# contourplot(surfacex,surfacey,surfacez,speciesdiff, 'speciesdiff')




file = open("newfile2.txt", "w")

file.write('surfacex='+str(surfacex)+'\n')
file.write('surfacey='+str(surfacey)+'\n')
file.write('surfacez='+str(surfacez)+'\n')
file.write('dropletx= '+str(dropletx)+'\n')
file.write('droplety= '+str(droplety)+'\n')
file.write('dropletz= '+str(dropletz)+'\n')
file.write('xposition='+str(xposition)+'\n')
file.write('yposition='+str(yposition)+'\n')
file.write('time_run_for= '+str(atomnumber)+'\n')
file.write('esame_for_species1= '+str(esamespecies1)+'\n')
file.write('esame_for_species2= '+str(esamespecies2)+'\n')
file.write('ediff_for_species1= '+str(ediffspecies1)+'\n')
file.write('ediff_for_species2= '+str(ediffspecies2)+'\n')
file.write('rdep1= '+str(dep1prefactor)+'\n')
file.write('rdep2= '+str(dep2prefactor)+'\n')
file.write('temp='+str(temp)+'\n')
file.write(str(processarray)+'\n')
file.write("---_%s seconds_---" % (end_time - start_time))


# file2 = open("spec1coords.csv", "w")
# file3 = open("spec2coords.csv", "w")
#
#
# for i in range(xposition,xposition+dropletx):
#     for j in range (yposition,yposition+droplety):
#          for k in range (0, wiretop[i][j]):
#              if (species1bools[i][j][k])==1:
#                  file2.write(str(i)+" "+str(j)+" "+str(k)+ "\n")
#              if (species2bools[i][j][k])==1:
#                  file3.write(str(i)+" "+str(j)+" "+str(k)+"\n")

np.savetxt('spec1coords.csv',species1bools.nonzero(),delimiter=', ', newline='\n')
np.savetxt('spec2coords.csv',species2bools.nonzero(),delimiter=', ',newline='\n')
np.savetxt('wiretop.csv',wiretop,delimiter=', ',newline='\n')
np.savetxt('sameneighbourtop.csv',sameneighbourtop,delimiter=', ',newline='\n')
np.savetxt('diffneighbourtop.csv',diffneighbourtop,delimiter=', ',newline='\n')


file2= open("samespeciesneighbours.csv", 'w')
file3= open('diffspeciesneighbours.csv', 'w')


for i in range(xposition,xposition+dropletx+1):
    for j in range(yposition, yposition+droplety+1):
        for k in range(0,wiretop[i][j]+1):

            file2.write(str(i)+','+str(j)+ ','+str(k)+','+str(samespeciesneighbours[i][j][k])+ '\n')
            file3.write(str(i)+','+str(j)+ ','+str(k)+','+str(diffspeciesneighbours[i][j][k])+ '\n')

a = izip(*csv.reader(open("spec1coords.csv", "rb")))
csv.writer(open("spec1coords.csv", "wb")).writerows(a)


a = izip(*csv.reader(open("spec2coords.csv", "rb")))
csv.writer(open("spec2coords.csv", "wb")).writerows(a)

# plotcrosssection(species1wire,species2wire, 3, 'cross-section at z=3')
# plotcrosssection(species1wire,species2wire, 5, 'cross-section at z=5')
#Plots everything
#plot3scatteronceattheend(species1wire,species2wire,surface3dbools)



#plothistonceattheend(surfacex,surfacey,surfaceint)
