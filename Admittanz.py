# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 21:57:20 2023

@author: Moussa Sarr Ndiaye
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 18:14:18 2022

@author: Moussa Sarr Ndiaye
"""


from cmath import pi, sqrt
#from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# from scipy.constants import speed_of_light
import os

from functools import partial
from multiprocessing import Pool, TimeoutError

import time
import csv 
import pandas as pd
# from scipy.optimize import curve_fit, minimize_scalar
from array import *
from scipy.constants import speed_of_light
# import control 
from shapely.geometry import LineString
import shapely.geometry
#import shapely.wkt
#from shapely.geometry import mapping, shape
# import control
import math




df1 = pd.read_csv('Z Parameter Plot 280.csv', sep=',', header=None)
#print(df)




"""print(df)"""

Frequences_1 = df1[1]
magnitude_1 = df1[2]
phase_1 = df1[3]

print(Frequences_1)

x_1=[]    #new array frequence 
y_1=[]    #new real part 
z_1=[]    #new imaginary part 



C = speed_of_light
#for i in range (0, 1000):
for i in range (0, 401):
    x_1.append(Frequences_1[i])
    y_1.append(magnitude_1[i])
    z_1.append(phase_1[i])

"""neu"""
x_11 = []
y_22 = []
Z_output_real_11 = []
Z_output_imag_11 = []  

Z_Load = np.zeros((len(x_1),1),  np.complex64)


for i in range (len(x_1)):
    
    
    Z_Load[i] = y_1[i] * np.exp(1j* math.radians(z_1[i]))
   
print('Zload =', Z_Load[1])
""" neu""" 



# maximum at resonance 
ymax = np.max(Z_Load.real)
xpos = np.where(Z_Load.real == ymax)
xmax = np.array(x)[xpos[0]]
print('resonant frequency and Impendance max = ', xmax,ymax)

Z_output = [] # content list of Z_input (Z_in real and Z_in imag)
Z_output_real = []   # list of real part 
Z_output_imag = []# list of imaginary part
B_L = np.zeros((len(x_1),1))
Z_in_serie = np.zeros((len(x_1),1), np.complex64)
Z_output2 = []

def Quarter_Wave_1(Z_l): 
    beta_quarter = []
    Z_opt = 50 
    C_impedanz = 50
    C = speed_of_light
    Z_0 = np.sqrt(Z_opt* ymax)                        # caracteristic impedance  
    print('Characteristic impedance =', Z_0)
    lamda = C/xmax  
    Beta_quarter = (2*np.pi)/ lamda   # calcul of Beta      
    length_wave = lamda/4*80 /90            # length of Stripline bei resonanz
    for idx,f in enumerate(x_1):
        Lamda_f = C/f
        Beta_quarter_l = ((2*np.pi)/Lamda_f) * length_wave
        B_L[idx] = Beta_quarter_l
        Zin_num = Z_0*(Z_l[idx] + 1j*Z_0*np.tan(Beta_quarter_l))      
        Zin_denom = (Z_0 + 1j*Z_l[idx]*np.tan(Beta_quarter_l))  
        Z_in_serie[idx] = Zin_num/Zin_denom                        # Input impedance
        
        
Quarter_Wave_1(Z_Load)
print('Shape of Z_in_serie =', Z_in_serie.shape)
#print(Z_in_serie)