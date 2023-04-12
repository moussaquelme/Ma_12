# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:45:09 2022

@author: Moussa Sarr Ndiaye
"""

from cmath import pi, sqrt
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.constants import speed_of_light
import os

from functools import partial
from multiprocessing import Pool, TimeoutError

from numba import jit
import time
import csv 
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar
from array import *
from scipy.constants import speed_of_light
import control 
from shapely.geometry import LineString
import shapely.geometry
import shapely.wkt
from shapely.geometry import mapping, shape
import math


df = pd.read_csv('Z1_7.in', sep=' ',header=None)
#print(df)


#Z_L = Z_L_Mag * exp (1i*deg2rad(Z_L_Phase)) 
#Z_L = Z_L_mag * np.exp(1j*math.radians(Z_L_phase/math.pi))
#print (math.radians(180 / math.pi))

"""print(df)"""
Frequences = df[0]
magnitude = df[1]
phase = df[2]


x=[]    #new array frequence 
y=[]    #new real part 
z=[]    #new imaginary part 

C = speed_of_light
for i in range (0, 1000):
    x.append(Frequences[i])
    y.append(magnitude[i])
    z.append(phase[i])
    
    
x_1 = []
y_2 = []
Z_output_real = []
Z_output_imag = []  

#def MagPhase_to_RealImag(L):
for i in range (0, 1000):
    #Z_Load = y[i] * np.exp(1j*math.radians(z[i]/math.pi))
    Z_Load = y[i] * np.exp(1j* math.radians(z[i]))
    x_1.append(Z_Load)
    Z_output_real.append(Z_Load.real)
    Z_output_imag.append(Z_Load.imag)
    print(Z_Load)
    
    
plt.plot(x,Z_output_real)
plt.plot(x,Z_output_imag)
plt.show()
    
    
    
    