# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:48:32 2022

@author: Clayton
"""
import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt
import random
import pandas as pd

l=5
kernel=np.ones(shape=(l,l,l))
kernel[int(np.ceil(l/2-1)),int(np.ceil(l/2-1))]=0

def rand_init(width, length, B_to_R,init_b,init_r):
    """ Random system initialisation.
    BLUE  =  0
    RED   =  1
    """
    
    population = length * width * width #population size
    
    blues = int(population *B_to_R) #number of blues
    reds = population - blues #number of reds
    M = np.full(length * width * width, init_b) #
    M[:int(reds)] = init_r
    np.random.shuffle(M)
    return M.reshape(length,width,width)

def compare(b_d,r_d,init_b,init_r):
    
    if b_d.sum()> r_d.sum():
        return  r_d,init_r,b_d,init_b
    
    elif r_d.sum()> b_d.sum():
        return  b_d,init_b, r_d,init_r
        
N = 50    # Grid will be N x N
L=100 #height/length of the film in the z-direction

SIM_T = 0.4   # Similarity threshold (that is 1-τ)


B_to_R = 0.5   # Ratio of blue to red people
init_b=1
init_r=0

M=rand_init(N, L, B_to_R, init_b, init_r)

# M=np.indices((L,N,N)).sum(axis=0)%2

kws = dict(mode='same')
# iterations=40

# for i in range(iterations):

#     b_neighs = convolve(M == init_b, kernel, **kws)
#     r_neighs = convolve(M == init_r, kernel, **kws)
#     neighs   = convolve(M !=-1,  kernel, **kws)
#     b_dissatisfied = (b_neighs / neighs < SIM_T) & (M == init_b)
#     b_satisfied=(b_neighs / neighs > SIM_T) & (M == init_b)
#     r_dissatisfied = (r_neighs / neighs < SIM_T) & (M == init_r)
#     r_satisfied=(r_neighs / neighs > SIM_T) & (M == init_r)
#     n_b_dissatisfied, n_r_dissatisfied = b_dissatisfied.sum(), r_dissatisfied.sum()
#     shorter,shorter_init,longer,longer_init=compare(b_dissatisfied,r_dissatisfied, init_b,init_r)
    
#     bdcoords=np.array(np.where(b_dissatisfied)).T
#     bscoords=np.array(np.where(b_satisfied)).T
#     rdcoords=np.array(np.where(r_dissatisfied)).T
#     rscoords=np.array(np.where(r_satisfied)).T
    
#     bdcoords=bdcoords[:shorter.sum()]
#     rdcoords=rdcoords[:shorter.sum()]
    
#     for coord in bdcoords:
#         M[coord[0],coord[1],coord[2]]=init_r
    
#     for coord in rdcoords:
#         M[coord[0],coord[1],coord[2]]=init_b
    

#     unique,counts=np.unique(M, return_counts=True)

# plt.imshow(M[50])
# from scipy import ndimage 

# def find_clusters(array):
#     clustered = np.empty_like(array)
#     unique_vals = np.unique(array)
#     cluster_count = 0
#     for val in unique_vals:
#         labelling, label_count = ndimage.label(array == val)
#         for k in range(1, label_count + 1):
#             clustered[labelling == k] = cluster_count
#             cluster_count += 1
#     return clustered, cluster_count

# clusters, cluster_count = find_clusters(M)    

# ones = np.ones_like(M, dtype=int)
# cluster_sizes = ndimage.sum(ones, labels=clusters, index=range(cluster_count)).astype(int)
# # com = ndimage.center_of_mass(ones, labels=clusters, index=range(cluster_count))
# # for i, (size, center) in enumerate(zip(cluster_sizes, com)):
# #     print("Cluster #{}: {} elements at {}".format(i, size, center))
# plt.figure()
# plt.plot(np.linspace(0,len(cluster_sizes)-1,len(cluster_sizes)),cluster_sizes)
# plt.ylim([0,10])
# SIZE = L*N*N
# SAMPLE_SIZE = int(SIZE*0.05)




# z, x, y = M.nonzero()

# idx = np.random.choice(np.arange(len(x)), SAMPLE_SIZE)


# fig = plt.figure(figsize=(N/10,L/10))
# ax = fig.add_subplot(111, projection='3d',box_aspect=(1,1,2))
# ax.scatter(x[idx], y[idx],z[idx],c=z[idx],alpha=1)
# # ax.scatter(x, y,z,c=z,alpha=1)
# plt.show()
# con_table=pd.DataFrame()
# plt.imshow(M[:,25,:])
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
    atom_stream=np.delete(atom_stream,randomlist)
    # atom_stream=np.random.choice(atom_stream, replace=False,size=int(atom_stream.size * ed))
    # atom_stream[indices]=0
    # atom_stream= np.random.choice(atom_stream, size=int(p*len(atom_stream)))
    block_num=34
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
p=B_to_R
# for i in np.linspace(1,class_size-1,class_size-1):
#     expected=expected*(B_to_R)/(1-(B_to_R))*(block_size+1-i)/i
#     expected_list=np.append(expected_list,expected)
prob_list=pd.DataFrame(0, index=np.arange(class_size),columns=['i','P','Exp'])
for i in np.arange(0,class_size):
    perm=np.math.factorial(int(block_size))/(np.math.factorial(int(i))*np.math.factorial(int(block_size-i)))
    print(i)
    P=perm*(p**i)*((1-p)**(block_size-i))
    prob_list['i'].loc[i]=i
    prob_list['P'].loc[i]=P
    expect=len(fullratiolist)*P
    # expect=expect*(p/(1-p))*(Nb+1-i)/i
    expected_list=np.append(expected_list,expect)
    prob_list['Exp'].loc[i]=expect
con_table=np.histogram(fullratiolist['nr'],bins=block_size)
# con_table=np.histogram(fullratiolist['nr'],bins=np.linspace(0,block_size,51))

plt.plot(prob_list['i'],prob_list['Exp'])

# plt.hist(fullratiolist['nr'],bins=np.linspace(0,block_size,51))
plt.xlabel('Box Size')
plt.ylabel('Counts')
lis=fullratiolist['nr']










