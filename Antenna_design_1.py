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
#from Admittanz.py import Quarter_Wave_1



#df = pd.read_csv('Z1_13.in', sep=' ',header=None)
#df = pd.read_csv('Zin_MagDeg_HFSS_40um_Inset.csv', sep=' ',header=None)
##df = pd.read_csv('Z Parameter Plot 270.csv', sep=',', header=None)
##df1 = pd.read_csv('Z Parameter Plot 280.csv', sep=',', header=None)
#print(df)

df = pd.read_csv('300.csv', sep=',',header=None)





"""print(df)"""
Frequences = df[1]
magnitude = df[2]
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

Z_output = []        # content list of Z_input (Z_in real and Z_in imag)
Z_output_real = []   # list of real part 
Z_output_imag = []   # list of imaginary part
B_L = np.zeros((len(x),1))
Z_in = np.zeros((len(x),1), np.complex64)
Z_output2 = []

def Quarter_Wave(Z_l): 
    beta_quarter = []
    Z_opt = 50 
    C_impedanz = 50
    C = speed_of_light
    Z_0 = np.sqrt(Z_opt* ymax)                        # caracteristic impedance  
    print('Characteristic impedance =', Z_0)
    lamda = C/xmax  
    Beta_quarter = (2*np.pi)/ lamda   # calcul of Beta      
    length_wave = lamda/4*80 /90            # length of Stripline bei resonanz
    #length_wave = lamda/4
    for idx,f in enumerate(x):
        Lamda_f = C/f
        Beta_quarter_l = ((2*np.pi)/Lamda_f) * length_wave
        B_L[idx] = Beta_quarter_l
        Zin_num = Z_0*(Z_l[idx] + 1j*Z_0*np.tan(Beta_quarter_l))      
        Zin_denom = (Z_0 + 1j*Z_l[idx]*np.tan(Beta_quarter_l))  
        Z_in[idx] = Zin_num/Zin_denom                        # Input impedance
        #Y_in[idx] = 1/ Z_in[idx]

Quarter_Wave(Z_Load)
print('Shape of Z_in =', Z_in.shape)

#Y_general = Y_in[idx] +  Y_in_serie[idx]
#print(Y_general)


def return_loss (Z_load, characteric_impedanz):
    S_11_of_f = np.abs((Z_load- characteric_impedanz)/ (Z_load + characteric_impedanz))
    S_11_dB = 20*np.log10(S_11_of_f)
    return S_11_of_f, S_11_dB
S_11_in, S_11_in_dB = return_loss(Z_in, 50)
S_11_load, S_11_load_dB = return_loss(Z_Load, 50)

# maximum S11_load 
ymax_load_dB = np.max(np.abs(S_11_load_dB))

print('ymax_load_dB=',ymax_load_dB)

# maximum S11_dB 
ymax_dB = np.max(np.abs(S_11_in_dB))

print('ymax_dB=',ymax_dB)






        


w = np.linspace(-10,-10,401) 
d = np.linspace(-10,-10,401)
     
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


ax1.plot(x,Z_Load.real)
ax1.plot(x,Z_Load.imag)
ax1.set_xlabel('Frequency in [Hz]', fontsize=6)
ax1.set_ylabel('IMPEDANCE [Ohm]', fontsize=6)
ax1.set_title('LOAD IMPEDANCE VS FREQUENCY', fontsize=8)
ax1.legend(['Real part', 'imaginary part'], fontsize=6)
ax1.annotate('Ymax Ohm',xy=(xmax, ymax), xytext=(xmax,ymax),fontsize=10)
#ax1.text(50e+9, 50, r'res. freq.:272.4 GHz , Imp 573 Ohm', fontsize=10)
#ax1.text(1.5e+10, 15, r'Max Imp: 9.37 Ohm', fontsize=10)

ax2.plot(x,S_11_load_dB)
ax2.set_xlabel('Frequency in [Hz]', fontsize=6)
ax2.set_ylabel('S11_Load [dB]', fontsize=6)
ax2.set_title('S11_Load in dB VS FREQUENCY', fontsize=8)
#intersection 
ax2.plot(x,d)
first_line = LineString(np.column_stack((x, d)))
second_line = LineString(np.column_stack((x, S_11_load_dB)))
intersection = first_line.intersection(second_line)

if intersection.geom_type == 'MultiPoint':
    BW_L1 = (LineString(intersection).xy[0][0])   #Bandwidth1
    BW_L2 = (LineString(intersection).xy[0][1])   #Bandwidth2
    BH_L1=(LineString(intersection).xy[1][0])   #Bandheight1
    BH_L2=(LineString(intersection).xy[1][1])   #Bandheight2
    print('BW_L1 =', BW_L1)
    print('BW_L2 =', BW_L2)
    print('BH_L1 =', BH_L1)
    print('BH_L2 =', BH_L2)
    ax2.plot(*LineString(intersection).xy, 'o')
    print('Bandwidth_load=',*LineString(intersection).xy)
    ax2.annotate('f1 =283.7 GHz ',xy=(BW_L1, BH_L1-0.8),fontsize=10)
    ax2.annotate('f2 =294.69 GHz',xy=(BW_L2, BH_L2),fontsize=10)
   
    
elif intersection.geom_type == 'Point':
    ax2.plot(*intersection.xy,x, 'o')
    print((*intersection.xy,x).coords)


ax3.plot(x,Z_in.real)
ax3.plot(x,Z_in.imag ,'r')
ax3.set_title('INPUT IMPEDANCE vs FREQUENCY', fontsize=8)
ax3.set_xlabel('FREQUENCY in [Hz]', fontsize=6)
ax3.set_ylabel('INPUT IMPEDANCE [Ohm]', fontsize=6)
ax3.legend(['Real part', 'imaginary part'], fontsize=6)

ax4.plot(x,S_11_in_dB)
#ax4.semilogx(x, S_11_in_dB)
ax4.set_xlabel('Frequency in [Hz]', fontsize=6)
ax4.set_ylabel('S11 in [dB]', fontsize=6)
ax4.set_title('S11 in dB VS FREQUENCY', fontsize=8)

ax4.plot(x,w)
first_line1 = LineString(np.column_stack((x, w)))
second_line1 = LineString(np.column_stack((x, S_11_in_dB)))
intersection = first_line1.intersection(second_line1)

if intersection.geom_type == 'MultiPoint':
    BW1 = (LineString(intersection).xy[0][0])   #Bandwidth1
    BW2 = (LineString(intersection).xy[0][1])   #Bandwidth2
    BH1=(LineString(intersection).xy[1][0])   #Bandheight1
    BH2=(LineString(intersection).xy[1][1])   #Bandheight2
    print('BW1 =', BW1)
    print('BW2 =', BW2)
    print('BH1 =', BH1)
    print('BH2 =', BH2)
   
    
    ax4.plot(*LineString(intersection).xy, 'o')
    print('Bandwidth_S11=',*LineString(intersection).xy)
    ##ax4.annotate('f1 =278.008 GHz ',xy=(BW1, -BH1),fontsize=10)
    ##ax4.annotate('f2 =279.57 GHz',xy=(BW2, -BH2),fontsize=10)
    ax4.annotate('f1 =284.76 GHz ',xy=(BW1, BH1-0.8),fontsize=10)
    ax4.annotate('f2 =289.88 GHz',xy=(BW2, BH2),fontsize=10)
elif intersection.geom_type == 'Point':
    ax4.plot(*intersection.xy,x, 'o')
    print((*intersection.xy,x).coords)
    ax4.annotate('f1 =284.76 GHz ',xy=(BW1, BH1-0.8),fontsize=10)
    ax4.annotate('f2 =289.88 GHz',xy=(BW2, BH2),fontsize=10)
   
##ax4.text(50e+9, -2, r'f1 = 278.008GHz , f2=279.57GHz', fontsize=10)


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