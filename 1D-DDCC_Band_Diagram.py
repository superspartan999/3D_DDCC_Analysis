# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:59:20 2019

@author: Clayton
"""

#Band Diagram
fig=plt.figure()    
i=-0.0
i=round(i,2)
print(i)
file='Project_1_10_doped'
filename = str(file)+'_result.out.vg_'+str(i)+'00-cb.res'
os.chdir(directory)
headerlist= ['Position', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
             'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
             'uEv','uEv2','effective trap','layernumber']
file=pd.read_csv(filename, sep="   ",header= None,engine='python')
file.columns=headerlist

plt.plot(file['Position']/1e-7,file['Ec'], label='Ec doped AlGaN',color='red')
#plt.plot(file['Position']/1e-7,file['Efp'], label='Hole Fermi Level',color='blue')
plt.plot(file['Position']/1e-7,file['Ev'], label='Ev doped AlGaN',color='orange')
plt.xlabel('z (nm)')
plt.ylabel('Energy (eV)')
plt.tight_layout()
plt.xlim(30,60)
#plt.ylim(-7,10)
plt.legend(framealpha=100)
plt.grid()
#plt.savefig('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\'+str(filename)+'Valence.png')
directory = 'D:\\1'
os.chdir(directory)
i=round(i,2)
print(i)
file='Project_1'
filename = str(file)+'_result.out.vg_'+str(i)+'00-cb.res'
os.chdir(directory)
headerlist= ['Position', 'Ec', 'Ev', 'Efn','Efp', 'n', 'p','Jn','Jp', 'Rad','Non-Rad','Rauger','RspPL', 'eb', 'ebh',\
             'generation','active dopant','impactG','1/uEc','1/uEv','1/uEhh','Electric field','mun','mup','uEc',\
             'uEv','uEv2','effective trap','layernumber']
file=pd.read_csv(filename, sep="   ",header= None,engine='python')
file.columns=headerlist

plt.plot(file['Position']/1e-7,file['Ec'], label='Ec undoped AlGaN', color='blue')
plt.plot(file['Position']/1e-7,file['Ev'], label='Ev undoped AlGaN', color='green')
plt.xlabel('z (nm)')
plt.ylabel('Energy (eV)')
plt.tight_layout()
#plt.xlim(0,1.6e-5)
#plt.ylim(-7,10)
plt.legend()
plt.grid()
#plt.savefig('C:\\Users\\Clayton\\Google Drive\Research\\Simulations\\'+str(filename)+'.png')