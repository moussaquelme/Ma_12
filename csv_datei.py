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

#df = pd.read_csv('Z1.in', delim_whitespace= True, header=None)
df = pd.read_csv('Z1_2.in', sep=' ',header=None)
#df = pd.read_csv('Z1.in',delimiter=' ')

"""print(df)"""
Frequences = df[0]
Real_part = df[1]
#print(Real_part)

#fig, ax =plt.subplots(figsize=(10,7))
plt.plot(Frequences, Real_part)
plt.xlabel('Frequencies in [Hz]', fontsize=14)
#plt.xlim([2.10e11, 3.10e11])
plt.ylabel('real part in [Ohm]', fontsize=14)
plt.title('Optimization 1 real part vs frequnecies', fontsize=14)

ymax = np.max(Real_part)
xpos = np.where(Real_part == ymax)
xmax = np.array(Frequences)[xpos]

#plt.annotate('local max', xy=(xmax, ymax), xytext=(xmax, ymax + 5), arrowprops=dict(facecolor='black'),)
#plt.annotate('local max', xy=(xmax, ymax))
#print(xmax, ymax)


#plt.show()
#Z_l = [38.9, 18.5, 40.52, 81.63, 131.38, 196.43, 266.6, 366.16, 405.97, 467.7, 532.19, 576.10]

#def Quarter_Wave(Z_0, Z_l, L):
def Quarter_Wave(Z_l):

    #Z_l = [38.9, 18.5, 40.52, 81.63, 131.38, 196.43, 266.6, 366.16, 405.97, 467.7, 532.19, 576.10]
    for Z_1 in enumerate (Z_all):
        Z_0 = 70
        Beta_quarter_l = np.pi/2
        Z_in = Z_0*((Z_l + 1j*Z_0*np.tan(Beta_quarter_l))/(Z_0 + 1j*Z_l*np.tan(Beta_quarter_l)))
        """print(Z_1)"""

    return Z_in
Z_all = np.array([38.9, 18.91, 18.5, 40.52, 81.63, 131.38, 196.43, 266.6, 366.16, 405.97, 467.7, 532.19, 576.10])
#Z_all = ["38.9", "18.91", "18.5", "40.52", "81.63", "131.38", "196.43", "266.6", "366.16", "405.97", "467.7", "532.19", "576.10"]
#print(Quarter_Wave(70, Z_l, 0.25 ))
print(Quarter_Wave(Z_all))
