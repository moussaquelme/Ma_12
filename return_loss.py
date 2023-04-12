# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 11:47:19 2022

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

df = pd.read_excel(r'result_1.4.xlsx', sheet_name='optimisation_13',header=0)

print(df)



Frequency = list(df['frequency'])
#Frequency = list(df.loc[3])

S_11dB = list(df['S_11dB'])
#S_11dB = list(df.loc[5])

Re = list(df['Re'])
#Re = list(df.loc[1])

Im = list(df['Im'])
#Im = list(df.loc[2])


"""ymax = -1*0.7*np.max(np.abs(S_11dB))
xpos = np.where(S_11dB == ymax)
xmax = np.array(Frequency)[xpos]
print(xmax,ymax)"""




fig, axs =plt.subplots(2)
#axs.figure(figsize=(10,10))
axs[0].plot(Frequency,S_11dB)
##axs[0].set_xlim(2.55e11, 3e11)
#axs[0].plot(plt.xlim(),[xmax,xmax],'r--')
#axs[0].plot([ymax ,ymax],plt.ylim(),'r--')

#axs[0].title("Excel sheet to Scatter Plot")
axs[1].plot(Frequency,Re)
axs[1].plot(Frequency,Im ,'r')
###axs[1].set_ylim(-100,80)
plt.show()


