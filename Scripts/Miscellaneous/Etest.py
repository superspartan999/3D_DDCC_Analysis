# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 01:28:03 2019

@author: Kun
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from math import floor, sqrt


mydf=pd.read_csv('E:\\Google Drive\\Research\\AlGaN Unipolar Studies\\10nmAlGaN\\p_structure_0.17_10nm-out.vg_0.00.vd_-0.20.vs_0.00.unified', delimiter=' ')
del mydf['Unnamed: 0']

    