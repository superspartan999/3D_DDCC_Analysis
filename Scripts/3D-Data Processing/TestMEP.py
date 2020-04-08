# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:03:42 2020

@author: Clayton
"""
from mep.optimize import ScipyOptimizer
from mep.path import Path
from mep.neb import NEB
from mep.models import LEPS
import matplotlib.pyplot as plt

leps = LEPS() # Test model 
op = ScipyOptimizer(leps) # local optimizer for finding local minima
x0 = op.minimize([1, 4], bounds=[[0, 4], [-2, 4]]).x # minima one
x1 = op.minimize([3, 1], bounds=[[0, 4], [-2, 4]]).x # minima two


path = Path.from_linear_end_points(x0, x1, 101, 1)  # set 101 images, and k=1
neb =NEB(leps, path) # initialize NEB
history = neb.run(verbose=True) # run
