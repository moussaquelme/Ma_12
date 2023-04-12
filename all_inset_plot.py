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
#import shapely.geometry
#import shapely.wkt
#from shapely.geometry import mapping, shape
# import control
import math




df = pd.read_csv('Z ParameterMA_60um.csv', sep=',', header=None)





"""print(df)"""
Frequences = df[2]
magnitude = df[4]
phase = df[3]




print(Frequences)

x=[]    #new array frequence 
y=[]    #new real part 
z=[]    #new imaginary part 

C = speed_of_light
#for i in range (0, 1000):
for i in range (0, 401):
    x.append(Frequences[i])
    y.append(magnitude[i])
    z.append(phase[i])

"""neu"""
x_1 = []
y_2 = []
Z_output_real_1 = []
Z_output_imag_1 = []  

Z_Load = np.zeros((len(x),1),  np.complex64)


for i in range (len(x)):
    
    
    Z_Load[i] = y[i] * np.exp(1j* math.radians(z[i]))
   
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
B_L = np.zeros((len(x),1))
Z_in = np.zeros((len(x),1), np.complex64)
Z_output2 = []



w = np.linspace(-6,-6,401) 

#d = np.linspace(-ymax_load_dB*0.5,-ymax_load_dB*0.5,1000)
#d = np.linspace(-ymax_load_dB*0.5,-ymax_load_dB*0.5,401) 

d = np.linspace(-6,-6,401)
     
#fig, (ax1) = plt.subplot()


plt.plot(x,Z_Load.real)
plt.plot(x,Z_Load.imag)
plt.xlabel('Frequency in [Hz]', fontsize=6)
plt.ylabel('IMPEDANCE [Ohm]', fontsize=6)
plt.title('LOAD IMPEDANCE VS FREQUENCY', fontsize=8)
plt.legend(['Real part', 'imaginary part'], fontsize=6)
plt.annotate('Ymax Ohm',xy=(xmax, ymax), xytext=(xmax,ymax),fontsize=10)
#ax1.text(50e+9, 50, r'res. freq.:272.4 GHz , Imp 573 Ohm', fontsize=10)
#ax1.text(1.5e+10, 15, r'Max Imp: 9.37 Ohm', fontsize=10)



plt.tight_layout()
plt.tight_layout()
plt.tight_layout()
plt.tight_layout()

plt.grid(b=None, which='major', axis='both')

plt.show()
#fig.savefig('opt13.png',dpi='figure',format=None, metadata=None,
        #bbox_inches=None, pad_inches=0.1,
        #facecolor='auto', edgecolor='auto',
        #backend=None)