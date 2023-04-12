# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 18:14:18 2022

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
import control
import math



df = pd.read_csv('Z1_3.in', sep=' ',header=None)
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

"""neu"""
x_1 = []
y_2 = []
Z_output_real_1 = []
Z_output_imag_1 = []  

#def MagPhase_to_RealImag(L):
for i in range (0, 1000):
    #Z_Load = y[i] * np.exp(1j*math.radians(z[i]/math.pi))
    Z_Load = y[i] * np.exp(1j* math.radians(z[i]))
    x_1.append(Z_Load)
    Z_output_real_1.append(Z_Load.real)
    Z_output_imag_1.append(Z_Load.imag)
    #print(Z_Load)  
""" neu""" 



# maximum 
ymax = np.max(Z_output_real_1)
xpos = np.where(Z_output_real_1 == ymax)
xmax = np.array(x)[xpos]
print(xmax,ymax)

Z_output = [] # content list of Z_input (Z_in real and Z_in imag)
Z_output_real = []   # list of real part 
Z_output_imag = []# list of imaginary part
B_L =[]
Z_output2 = []

def Quarter_Wave(Z_l): 
    beta_quarter = []
    Z_opt = 70 
    C_impedanz = 50
    C = speed_of_light
    Z_0 = np.sqrt(Z_opt* ymax)                        # caracteristic impedance  
    print(Z_0)
    lamda = C/xmax  
    Beta_quarter = (2*np.pi)/ lamda   # calcul of Beta      
    length_wave = lamda/4             # length of Stripline bei resonanz 
    for f in (x):
        Lamda_f = C/f
        Beta_quarter_l = ((2*np.pi)/Lamda_f) * length_wave   # B*l
        B_L.append(Beta_quarter_l)
        Zin_num = Z_0*(Z_l + 1j*Z_0*np.tan(Beta_quarter_l))      
        Zin_denom = (Z_0 + 1j*Z_l*np.tan(Beta_quarter_l))  
        Z_in = Zin_num/Zin_denom                        # Input impedance 
        Z_input = (Z_in.real, Z_in.imag)
        Z_output.append(np.abs(Z_in))
        Z_output2.append(Z_in)
        Z_output_real.append(Z_in.real)
        Z_output_imag.append(Z_in.imag)
Quarter_Wave(ymax)
Z_output2 = np.array(Z_output2)
print(Z_output2.shape)


list_S_11 = []
list_S11_dB = []
S_11_dB = []
S_11_of_f = []
def return_loss (characteric_impedanz):
    #for C in Z_output2:
    S_11_of_f = np.abs((Z_output2- characteric_impedanz)/ (Z_output2 + characteric_impedanz))
    S_11_dB = 20*np.log10(S_11_of_f)
    #list_S_11.append(S_11_of_f)
    #list_S11_dB.append(S_11_dB)
return_loss(50)
#print(S_11_of_f.shape)

        
w = np.linspace(-40,-40,1000)      
#fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


ax1.plot(x,Z_output_real_1)
ax1.plot(x,Z_output_imag_1)
ax1.set_xlabel('Frequency in [Hz]', fontsize=6)
ax1.set_ylabel('IMPEDANCE [Ohm]', fontsize=6)
ax1.set_title('LOAD IMPEDANCE VS FREQUENCY', fontsize=6)
ax1.legend(['Real part', 'imaginary part'], fontsize=6)
ax1.annotate('9.37 Ohm',xy=(xmax, ymax), xytext=(xmax,ymax),fontsize=10)
#ax1.annotate('maximum impedance resonance frequency', xy=(xmax, ymax))
ax1.text(1.5e+10, 10, r'res. freq.:278.3 GHz ', fontsize=10)
ax1.text(1.5e+10, 15, r'Max Imp: 9.37 Ohm', fontsize=10)

ax2.plot(x,S_11_dB)
ax2.set_xlabel('Frequency in [Hz]', fontsize=6)
ax2.set_ylabel('S11 [dB]', fontsize=6)
ax2.set_title('S11 in dB VS FREQUENCY', fontsize=6)
ax2.plot(x,w)
first_line = LineString(np.column_stack((x, w)))
second_line = LineString(np.column_stack((x, list_S11_dB)))
intersection = first_line.intersection(second_line)

if intersection.geom_type == 'MultiPoint':
    ax2.plot(*LineString(intersection).xy, 'o')
    print(*LineString(intersection).xy)
    ax2.annotate('f1 =130 GHz ',xy=(130661222109.19221, -34),fontsize=10)
    ax2.annotate('f2 =142 GHz, S11:-40dB ',xy=(142046483078.92752, -40),fontsize=10)
    ax2.annotate('f3 =338 GHz',xy=(330048739090.9928, -44),fontsize=10)
    ax2.annotate('f4 =342 GHz',xy=(342803606899.2599, -40),fontsize=10)
   
    
elif intersection.geom_type == 'Point':
    ax2.plot(*intersection.xy,x, 'o')
    print((*intersection.xy,x).coords)


ax3.plot(x,Z_output_real)
ax3.plot(x,Z_output_imag ,'r')
ax3.set_title('INPUT IMPEDANCE vs FREQUENCY', fontsize=6)
ax3.set_xlabel('FREQUENCY in [Hz]', fontsize=6)
ax3.set_ylabel('INPUT IMPEDANCE [Ohm]', fontsize=6)
ax3.legend(['Real part', 'imaginary part'], fontsize=6)

#ax4.plot(x,list_S11_dB)
ax4.semilogx(x, S_11_dB)
ax4.set_xlabel('Frequency in [Hz]', fontsize=6)
ax4.set_ylabel('S11 [dB]', fontsize=6)
ax4.set_title('S11 in dB VS FREQUENCY', fontsize=6)


plt.tight_layout()
plt.tight_layout()
plt.tight_layout()
plt.tight_layout()

ax1.grid(b=None, which='major', axis='both')
ax2.grid(b=None, which='major', axis='both')
ax3.grid(b=None, which='major', axis='both')
ax4.grid(b=None, which='major', axis='both')
plt.show()
#fig.savefig('opt13.png',dpi='figure',format=None, metadata=None,
        #bbox_inches=None, pad_inches=0.1,
        #facecolor='auto', edgecolor='auto',
        #backend=None)


#plt.plot(x, f, '-')
#plt.plot(x, g, '-')

#idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
#plt.plot(x[idx], f[idx], 'ro')
#plt.show()












