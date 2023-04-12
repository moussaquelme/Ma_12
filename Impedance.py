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


df = pd.read_csv('Z1_3.in', sep=' ',header=None)


"""print(df)"""
Frequences = df[0]
Real_part = df[1]

freq = (Frequences[700], Frequences[800])
rea  = (Real_part[700], Real_part[800])

x=[]
y=[]

for i in range (700, 800):
    x.append(Frequences[i])
    y.append(Real_part[i])



#print(Frequences)
#print(Real_part)

""" new Array"""
#print(x)
#print(y)

#fig, ax =plt.subplots(figsize=(10,7))
plt.plot(x, y)


plt.xlabel('Frequencies in [Hz]', fontsize=14)
#plt.xlim(2.6e+11, 3e+11)

plt.ylabel('real part in [Ohm]', fontsize=14)
plt.title('Optimization 1 real part vs frequnecies', fontsize=14)





#ymax = np.max(Real_part)
ymax = np.max(y)
xpos = np.where(y == ymax)
xmax = np.array(x)[xpos]
print(xmax,ymax)
#Real_part[996]
#Real_part[500]


plt.show()

"""

def Quarter_Wave(*Z_l):
   
    #Z_0 = 70
  
    Z_opt = 70 
    
    Beta_quarter = 2*np.pi
    length_wave = 1/4
    Beta_quarter_l = Beta_quarter*length_wave
    Z_output = [] 
    Z_last = []  

    for Z_l in (Z_all):
        Z_last.append(Z_l)
        #Z_in = Z_0*((Z_l + 1j*Z_0*np.tan(Beta_quarter_l))/(Z_0 + 1j*Z_l*np.tan(Beta_quarter_l)))
        Z_0 = np.sqrt(Z_opt*Z_l) 
        Z_in = (Z_0 * Z_0) / Z_l
        #print(Z_l)
        Z_output.append(Z_in)

    return Z_last, Z_output

Z_all = [38.9, 18.91, 18.5, 40.52, 81.63, 131.38, 196.43, 266.6, 366.16, 405.97, 467.7, 532.19, 576.10]
print(Quarter_Wave(Z_all))

def S_parameter(Z_l,Z_S):
    #S_all = []
    Z_0 = 70
    Beta_quarter = 2*np.pi
    length_wave = 1/4
    Beta_quarter_l = Beta_quarter*length_wave
    #for Z_l in (Z_all_1):
    num = ((np.power(Z_0,2) - np.power(Z_S,2))*np.sin(Beta_quarter_l))+((Z_0*(Z_l - Z_S))*np.cos(Beta_quarter_l))
    denom = ((Z_0*(Z_l + Z_S))*np.cos(Beta_quarter_l))+((np.power(Z_0,2) + Z_S*Z_l)*np.sin(Beta_quarter_l))
    S_11 = num / denom
    #S_all.append(S_11)
    #return S_all
    return S_11
#Z_all_1 = [38.9, 18.91, 18.5, 40.52, 81.63, 131.38, 196.43, 266.6, 366.16, 405.97, 467.7, 532.19, 576.10]
#Z_out = np.array(125.96401028277636, 259.12215758857747, 264.86486486486484, 120.92793682132279, 60.026950875903474, 37.296392144923125, 24.94527312528636, 18.37959489872468, 13.382128031461654, 12.069857378624036, 10.476801368398545, 9.207238016497866, 8.50546780072904)
print(S_parameter(576.1, 8.51 ))
"""