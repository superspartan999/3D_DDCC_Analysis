# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 21:35:58 2022

@author: me_hi
"""
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.stats import chisquare
import numpy as np

N=100
L=1000
i_Ga=0
i_In=1
M = np.full(L*N*N, i_Ga) #

l=4
kernel=np.ones(shape=(l,l,l))
kernel[int(np.ceil(l/2-1)),int(np.ceil(l/2-1)),int(np.ceil(l/2-1))]=0

tot=M.size
composition=0.3
uniformity=0.8

init_In=int(composition*uniformity*tot)

M[:int(init_In)] = i_In

np.random.shuffle(M)

M=M.reshape(L,N,N)

remainder=int(composition*tot-init_In)

# Ga=M==i_Ga


# Ga_coords=np.array(np.where(M==i_Ga)).T
# In_coords=np.array(np.where(M==i_In)).T    

# Gadf=pd.DataFrame(Ga_coords, columns=['x','y','z'])
kws = dict(mode='same')
Ga_neighs = convolve(M == i_Ga, kernel, **kws)
In_neighs = convolve(M == i_In, kernel, **kws)
Ga_neighs[Ga_neighs<0.00001]=0
In_neighs[In_neighs<0.00001]=0
neighs = convolve(M != 1000, kernel, **kws)
spatial_probability=In_neighs/neighs
spatial_probability[spatial_probability<0.0001]=0

isGa=(M==i_Ga)
Ga_coords=np.array(np.where(M==i_Ga)).T
Gadf=pd.DataFrame(Ga_coords, columns=['x','y','z'])
spatial_list=np.zeros(shape=len(Gadf.index))

for i, Ga_coord in enumerate(Ga_coords):
      spatial_list[i]=spatial_probability[Ga_coord[0],Ga_coord[1],Ga_coord[2]]

    
spatial_list=spatial_list*1000000
# for i, Ga_coord in enumerate(Ga_coords):
#      spatial_list[i]=spatial_probability[Ga_coord[0],Ga_coord[1],Ga_coord[2]]


# # Gadf['weights']=spatial_list

coord_list=np.random.choice(np.arange(len(Ga_coords)),p=spatial_list/np.sum(spatial_list),size=remainder,replace=False)
remainder_list=Ga_coords[coord_list]
# remainder_list=np.unique(remainder_list,axis=0)[:remainder]
# np.random.shuffle(remainder_list)
i=0
j=0
for switch in remainder_list:
    if M[switch[0],switch[1],switch[2]]==0:
        M[switch[0],switch[1],switch[2]]=1
        # print(i)
        # i+=1
    # else:
    #     print(j)
    #     j+=1
    
# In_coords=np.array(np.where(M==i_In)).T    
# plt.figure()
# plt.imshow(M[:,50,:])
# Indf=pd.DataFrame(In_coords, columns=['x','y','z'])

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# z,x,y=M.nonzero()
# ax.scatter(x, y, -z, zdir='z', c= 'red')
# # ax.scatter(pos[0], pos[1], pos[2])
# plt.show()
init_b=i_Ga
init_r=i_In
fullratiolist=pd.DataFrame(columns=['nb','nr','nz'])
for layer in M:
    bool_arr = np.full((N,N), False)
    
    # create a list of randomly picked indices, one for each row
    idx = np.random.randint(N, size=N)
    
    # replace "False" by "True" at given indices
    bool_arr[range(N), idx] = True
    # Array for random sampling
    # sample_arr = [True, False]
    ed=2/3
    # bool_arr=np.random.choice(a=[True, False], size=(N, N), p=[ed, 1-ed])
    # # # Create a 2D numpy array or matrix of 3 rows & 4 columns with random True or False values
    # sample_arr=[True,False]
    # bool_arr = np.random.choice(sample_arr, size=(N,N))

    
    
    random_sample=layer.copy()
    sample_coords=np.array(np.where(bool_arr==False)).T
    
    # for coord in sample_coords:
    #     random_sample[coord[0],coord[1]]=0
        
    atom_stream=random_sample.flatten()
    # atom_stream=atom_stream[:-1]
    randomlist=random.sample(range(atom_stream.size), int(atom_stream.size*(1-ed))+1)
    # atom_stream=np.delete(atom_stream,randomlist)
    # atom_stream=np.random.choice(atom_stream, replace=False,size=int(atom_stream.size * ed))
    # atom_stream[indices]=0
    # atom_stream= np.random.choice(atom_stream, size=int(p*len(atom_stream)))
    block_num=200
    block_list=np.split(atom_stream,block_num)
    block_size=len(block_list[1])   
    ratiolist=pd.DataFrame({'nb':np.zeros(len(block_list)),'nr':np.zeros(len(block_list)),'nz':np.zeros(len(block_list))})
    
    for i,block in enumerate(block_list):
        nb=np.count_nonzero(block==init_b)
        nr=np.count_nonzero(block==init_r)
        nz=np.count_nonzero(block)
        nb_ratio=nb/block_size
        nr_ratio=nr/block_size
        n_non_zero_ratio=nz/block_size
        ratiolist['nb'].iloc[i]=nb
        ratiolist['nr'].iloc[i]=nr
        ratiolist['nz'].iloc[i]=nz
    
    fullratiolist=pd.concat([fullratiolist,ratiolist])

fullratiolist['nbratio']=fullratiolist['nb']/(fullratiolist['nb']+fullratiolist['nr'])
fullratiolist['nrratio']=fullratiolist['nr']/(fullratiolist['nb']+fullratiolist['nr'])
fullratiolist['nzratio']=fullratiolist['nz']/(fullratiolist['nb']+fullratiolist['nr'])
# expected=len(block_list)*(B_to_R)**(block_size)
class_size=block_size
expected_list=np.array([])
p=composition
# for i in np.linspace(1,class_size-1,class_size-1):
#     expected=expected*(B_to_R)/(1-(B_to_R))*(block_size+1-i)/i
#     expected_list=np.append(expected_list,expected)
prob_list=pd.DataFrame(0, index=np.arange(class_size+1),columns=['count','P','Exp'])
for i in np.arange(0,class_size+1):
    perm=np.math.factorial(int(block_size))/(np.math.factorial(int(i))*np.math.factorial(int(block_size-i)))

    P=perm*(p**i)*((1-p)**(block_size-i))
    prob_list['count'].loc[i]=i
    prob_list['P'].loc[i]=P
    expect=len(fullratiolist)*P
    # expect=expect*(p/(1-p))*(Nb+1-i)/i
    expected_list=np.append(expected_list,expect)
    prob_list['Exp'].loc[i]=expect
con_table=np.histogram(fullratiolist['nr'],bins=block_size)
# con_table=np.histogram(fullratiolist['nr'],bins=np.linspace(0,block_size,51))



num_of_atom, frequency=np.unique(fullratiolist['nr'].values,return_counts=True)
combined_list=np.vstack((num_of_atom, frequency)).T
plt.plot(prob_list['count'],prob_list['Exp'])
plt.bar(num_of_atom, frequency)
# plt.hist(fullratiolist['nr'],bins=np.linspace(0,block_size,51))
plt.xlabel('Box Size')
plt.ylabel('Counts')
lis=fullratiolist['nr']
prob_list['Sum']=prob_list['Exp'].cumsum()
prob_list['Reverse Sum'] = prob_list.loc[::-1, 'Exp'].cumsum()[::-1]
full_list_atoms=np.zeros(shape=class_size+1)
for i,element in enumerate(full_list_atoms):
    for element_atom in combined_list:
        if i == element_atom[0]:
            full_list_atoms[i]=element_atom[1]
    #     print(i)


#make a dataframe of observed atoms   
observed_atoms=pd.DataFrame(np.vstack((np.arange(0,len(full_list_atoms)),full_list_atoms)).T,columns=['num of atoms','frequency'])
#find cumulative sum and make a new column for it
observed_atoms['Sum']=observed_atoms['frequency'].cumsum()

#find a reversed cumulative sum and make a new column for it
observed_atoms['Reverse Sum'] = observed_atoms.loc[::-1, 'frequency'].cumsum()[::-1]

#returned a sliced probability dataframe with cumulative sum thats higher than 5. Chi-square analysis needs
#column entry of value more than 5
prob_list=prob_list.loc[prob_list['Sum']>5]

#reset index
prob_list=prob_list.reset_index(drop=True)

#replace expected value in the first row with the sum of the first n rows,
prob_list['Exp'].iloc[0]=prob_list['Sum'].iloc[0]

#
prob_list=prob_list.loc[prob_list['Reverse Sum']>5]
prob_list['Exp'].iloc[-1]=prob_list['Reverse Sum'].iloc[-1]


observed_atoms=observed_atoms.loc[0:prob_list['count'].iloc[-1]]
observed_atoms=observed_atoms.loc[prob_list['count'].iloc[0]:]
observed_atoms['frequency'].iloc[0]=observed_atoms['Sum'].iloc[0]
observed_atoms['frequency'].iloc[-1]=observed_atoms['Reverse Sum'].iloc[-1]
observed_atoms=observed_atoms.reset_index(drop=True)


# observed_list=full_list_atoms[prob_list['count'].iloc[0]:prob_list['count'].iloc[-1]]
# if len(frequency) < class_size:

#     full_list_atoms=np.append(np.concatenate((np.arange(0,num_of_atom[0]),num_of_atom)),np.arange(num_of_atom[-1],class_size+1))
    
#     full_frequency=np.append(np.concatenate((np.zeros(shape=int(num_of_atom[0])).astype(int),frequency)),np.zeros(shape=class_size-int(num_of_atom[-1])))

#     frequency=full_frequency
chi,p_value=chisquare(observed_atoms['frequency'].values, f_exp=prob_list['Exp'].values)

chi_calculate=np.sum((observed_atoms['frequency'].values-prob_list['Exp'].values)**2/prob_list['Exp'].values)