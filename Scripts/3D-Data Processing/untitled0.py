# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:50:19 2022

@author: Clayton
"""

import pandas as pd
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from scipy.spatial import KDTree
from scipy.spatial import cKDTree
N=100

p=0.8

directorylist=['G:\\My Drive\\Research\\Transport Structure 2020\\071420AA - Reference']
# directory='G:\My Drive\Research\Transport Structure 2020\\072120AA - 15nm InGaN'
# directory='G:\My Drive\Research\Transport Structure 2020\\072120AB - 30nm InGaN'

# directorylist=['G:\My Drive\Research\Transport Structure 2020\\071420AA - Reference','G:\My Drive\Research\Transport Structure 2020\\072120AA - 15nm InGaN','G:\My Drive\Research\Transport Structure 2020\\072120AB - 30nm InGaN']
# directorylist=['G:\My Drive\Research\Transport Structure 2020\\071420AA - Reference','G:\My Drive\Research\Transport Structure 2020\\072320AB  - 1 x 5nm QW','G:\My Drive\Research\Transport Structure 2020\\072420AC - 3 x 5nm QW']
# directorylist=['G:\My Drive\Research\Transport Structure 2020\\071420AA - Reference']
#               'C:\\Users\\Clayton\\Google Drive\\Research\\Transport Structure 2020\\072120AA - 15nm InGaN',
#               'C:\\Users\\Clayton\\Google Drive\\Research\\Transport Structure 2020\\072120AB - 30nm InGaN']
#directorylist=['C:\\Users\\Clayton\\Google Drive\\Research\\Transport Structure Project\\Tunnel Junction IV\\Batch 4\\AlGaN Comparison\\110819AA']

filedict={}
for directory in directorylist:
    os.chdir(directory)


random_x_list = np.random.randint(0,100,size=N)
random_y_list = np.random.randint(0,100,size=N)


#def find_neighbours(alloy,x,y):
#    
#    

        PtoA=2/((diameter[0])*1e-4/2)
    
        
        PtoADict[PtoA] = pd.read_csv(key+'umr.csv')
        PtoADict[PtoA] = PtoADict[PtoA].dropna()
        
        DataFrameDict[key]['I']=DataFrameDict[key]['I']/area
        PtoADict[PtoA]['I']=PtoADict[PtoA]['I']/area
        x=DataFrameDict[key]['V']
        y=abs(DataFrameDict[key]['I'])
        plt.semilogy(x,y,label=key+ r' $\mu$m')
#        plt.xlim(0,7)
        list_IV.append(DataFrameDict[key])
    
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current Density (A/cm$^2$)')
        plt.grid()
#    
    fig=plt.figure(2)
    
    maxvolt=[]
    minvolt=[]
    
    for key in PtoADict.keys():
        
        maxvolt.append(PtoADict[key]['V'].iloc[0])
        minvolt.append(PtoADict[key]['V'].iloc[-1])     
        
    contribution=[]
        
    voltages=np.arange(-4,3,0.1)
    voltages=np.delete(voltages,np.where(voltages == 0))
    
    fits={}
    mc=pd.DataFrame(columns=['V','Slope','Intercept','r2'])
#    rr=pd.DataFrame(columns=['Diameter','Ratio'])
    
    for volt in voltages:
          jplist=[]
          ptoalist1=[]
          jperimeterlist=[]
          jlist=[]
        
          for key in PtoADict.keys():
              jp=PtoADict[key].iloc[(PtoADict[key]['V']-volt).abs().argsort()[:1]]['I'].values[0]
              jplist.append(jp)
              ptoalist1.append(key)
              
         
          fits[volt]= pd.DataFrame({'j':jplist,'p/a':ptoalist1})
          
          
          p=np.polyfit(ptoalist1,jplist,1)
          jperimeterlist=np.array(ptoalist1)*p[0]
          jlist=jperimeterlist+p[1]

          f=np.poly1d(p)
          x=np.linspace(ptoalist1[-1],ptoalist1[0],np.size(jplist))
          y=f(x)
          ratio=np.array(jperimeterlist)/np.array(jplist)
          diameter=np.array(4/np.array(ptoalist1).astype(int)/1e-4).astype(int)
          plt.plot(x,y, linestyle='dashed')
          plt.scatter(ptoalist1,jplist,label=str(volt)+' V')
          regression=r2_score(np.flip(jplist),f(x))
#          plt.scatter(ptoalist1,jlist)
          plt.grid()
          # plt.scatter(radius,ratio)
#
          mcrow={'V':volt,'Slope':p[0], 'Intercept':p[1],'r2':regression}
          dr=pd.DataFrame({'Diameter':diameter,'Ratio':ratio})
          mc=mc.append(mcrow, ignore_index = True)
#          print(dr['Diameter'].iloc[-1])
          size=100
          contribution.append(dr['Ratio'].loc[dr['Diameter']==size].values[0])
    plt.title('J vs P/A')
    plt.xlabel('P/A (cm$^{-1}$)')
    plt.ylabel('J (A/cm$^2$')
#          rr=rr.append(drrow, ignore_index= True)

    fig=plt.figure(7)
    plt.scatter(voltages,np.array(contribution)*100, label=size)
    plt.title('Percentage Contribution of J$_{perimeter}$ for 200 micron device')
    plt.xlabel('Voltage (V)')
    plt.yticks(np.arange(0, 100, step=10))
    plt.ylim(0,100)
    plt.xlim(-4,2.5)
    plt.ylabel('Percentage Contribution of J$_{perimeter}$ (%)')
    plt.grid()
    name=directory
    name=name.replace('G:\My Drive\Research\Transport Structure 2020\\', '')
       
    colors = plt.cm.jet(np.linspace(0,1,np.size(voltages)+1))
    poslist=[x for x in fits.keys() if x > 0]
    neglist=[x for x in fits.keys() if x < 0]
    neglist.sort(reverse=True)
    pos={}
    neg={}
    for i,key in enumerate(fits.keys()):   
        if key<0:
            neg[key]=fits[key]
    for i,key in enumerate(fits.keys()):  
        if key>0:
            pos[key]=fits[key]
    fig=plt.figure(3)  
    # neg=sorted(neg.items(),key=operator.itemgetter(1),reverse=True)
    for i,key in enumerate(poslist):   
            plt.plot(pos[key]['p/a'],abs(pos[key]['j']),label=str(key)+' V',color=colors[i])
            
    # for i,key in enumerate(neglist):   
    #         plt.semilogy(neg[key]['p/a'],abs(neg[key]['j']),label=str(key)+' V',linestyle='dashed',color=colors[i])
            
    plt.title('J vs P/A')
    plt.xlabel('P/A (cm$^{-1}$)')
    plt.ylabel('J (A/cm$^2$')
    fig=plt.figure(4)
    plt.plot(mc['V'],mc['Slope'],label=directory)
    plt.title('J Perimeter vs Volt')  
    plt.grid()
    plt.xlabel('Applied Bias (V)')
    plt.ylabel('$J_{perimeter}$ (A/cm)')    
    
    fig=plt.figure(5)
    
    plt.plot(mc['V'],mc['Intercept'],label=directory)   
    plt.title('J Diode vs Volt')  
    plt.xlabel('Applied Bias (V)')
    plt.grid()
    plt.ylabel('$J_{diode}$ (A/cm$^{2}$)') 
    
    filedict[name]=mc
#    

    
