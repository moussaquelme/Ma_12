# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:55:01 2023

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
#from Admittanz.py import Quarter_Wave_1




df = pd.read_csv('310.csv', sep=',', header=None)

hf = pd.read_csv('300.csv', sep=',', header=None)

sf = pd.read_csv('290.csv', sep=',', header=None)




"""print(df)"""
Frequences = df[1]
magnitude = df[2]
phase = df[3]

""" (hf)"""
mag = hf[2]
phi = hf[3]

""" (sf)"""
mag_sf = sf[2]
phi_sf = sf[3]


print(Frequences)


x=[]    #new array frequence 
y=[]    #new real part 
z=[]    #new imaginary part 

u =[]
v =[]

a = []
b = []




C = speed_of_light
#for i in range (0, 1000):
for i in range (0, 401):
    
    x.append(Frequences[i])
    y.append(magnitude[i])
    z.append(phase[i])
    
    u.append(mag[i])
    v.append(phi[i])
    
    a.append(mag_sf[i])
    b.append(phi_sf[i])
    


Z_Load = np.zeros((len(x),1),  np.complex64)

Z_Load_2 = np.zeros((len(x),1),  np.complex64)

Z_Load_3 = np.zeros((len(x),1),  np.complex64)


for i in range (len(x)):
    
    
    Z_Load[i] = y[i] * np.exp(1j* math.radians(z[i]))
    
    Z_Load_2[i] = u[i] * np.exp(1j* math.radians(v[i]))
    
    Z_Load_3[i] = a[i] * np.exp(1j* math.radians(b[i]))
   
print('Zload and Zload_2=', Z_Load[1],Z_Load_2[i], Z_Load_3[i])
""" neu""" 



# maximum at resonance for opt 1
ymax = np.max(Z_Load.real)
xpos = np.where(Z_Load.real == ymax)
xmax = np.array(x)[xpos[0]]
print('resonant frequency and Impendance max = ', xmax,ymax)


""" for optimization 2"""
ymax_2 = np.max(Z_Load_2.real)
xpos_2 = np.where(Z_Load_2.real == ymax_2)
xmax_2 = np.array(x)[xpos_2[0]]


print('resonant frequency and Impendance max = ', xmax_2,ymax_2)




""" for optimization 3"""
ymax_3 = np.max(Z_Load_3.real)
xpos_3 = np.where(Z_Load_3.real == ymax_3)
xmax_3 = np.array(x)[xpos_3[0]]

print('resonant frequency and Impendance max = ', xmax_3,ymax_3)




Z_output = []        # content list of Z_input (Z_in real and Z_in imag)
Z_output_real = []   # list of real part 
Z_output_imag = []   # list of imaginary part


B_L = np.zeros((len(x),1))
B_L_2 = np.zeros((len(x),1))
B_L_3 = np.zeros((len(x),1))

Z_in = np.zeros((len(x),1), np.complex64)
Z_in_2 = np.zeros((len(x),1), np.complex64)
Z_in_3 = np.zeros((len(x),1), np.complex64)

Z_output2 = []

def Quarter_Wave(Z_l, Z_l_2, Z_l_3): 
    beta_quarter = []
    Z_opt = 70 
    C_impedanz = 50
    C = speed_of_light
    Z_0 = np.sqrt(Z_opt* ymax)                        # caracteristic impedance opt 1
    
    Z_0_2 = np.sqrt(Z_opt* ymax_2)                        # caracteristic impedance opt 2
    
    Z_0_3 = np.sqrt(Z_opt* ymax_3)                        # caracteristic impedance opt 3
    
    print('Characteristic impedance =', Z_0, Z_0_2, Z_0_3)
    
    lamda = C/xmax
    lamda_2 = C/xmax_2 
    lamda_3 = C/xmax_3
    
    Beta_quarter = (2*np.pi)/ lamda   # calcul of Beta
    Beta_quarter_2 = (2*np.pi)/ lamda_2   # calcul of Beta
    Beta_quarter_3 = (2*np.pi)/ lamda_3   # calcul of Beta
      
    length_wave = lamda/4*80 /90            # length of Stripline bei resonanz
    length_wave_2 = lamda_2/4*80 /90            # length of Stripline bei resonanz
    length_wave_3 = lamda_3/4*80 /90            # length of Stripline bei resonanz
    
    
    for idx,f in enumerate(x):
        Lamda_f = C/f
        Lamda_f_2 = C/f
        Lamda_f_3 = C/f
        
        Beta_quarter_l = ((2*np.pi)/Lamda_f) * length_wave
        Beta_quarter_l_2 = ((2*np.pi)/Lamda_f_2) * length_wave_2
        Beta_quarter_l_3 = ((2*np.pi)/Lamda_f_3) * length_wave_3
        
        B_L[idx] = Beta_quarter_l
        B_L_2[idx] = Beta_quarter_l_2
        B_L_3[idx] = Beta_quarter_l_3
        
        Zin_num = Z_0*(Z_l[idx] + 1j*Z_0*np.tan(Beta_quarter_l))
        Zin_num_2 = Z_0_2*(Z_l_2[idx] + 1j*Z_0_2*np.tan(Beta_quarter_l_2))
        Zin_num_3 = Z_0_3*(Z_l_3[idx] + 1j*Z_0_3*np.tan(Beta_quarter_l_3)) 
        
        
        Zin_denom = (Z_0 + 1j*Z_l[idx]*np.tan(Beta_quarter_l))
        Zin_denom_2 = (Z_0_2 + 1j*Z_l_2[idx]*np.tan(Beta_quarter_l_2))
        Zin_denom_3 = (Z_0_3 + 1j*Z_l_3[idx]*np.tan(Beta_quarter_l_3))
        
        Z_in[idx] = Zin_num/Zin_denom                        # Input impedance
        Z_in_2[idx] = Zin_num_2/Zin_denom_2                        # Input impedance opt 2
        Z_in_3[idx] = Zin_num_3/Zin_denom_3                        # Input impedance opt 2
        #Y_in[idx] = 1/ Z_in[idx]

Quarter_Wave(Z_Load, Z_Load_2, Z_Load_3)
print('Shape of Z_in =', Z_in.shape, Z_in_2.shape, Z_in_3.shape)

#Y_general = Y_in[idx] +  Y_in_serie[idx]
#print(Y_general)

Y_adm = []
Z_adm = []

#def Admittanz (H1, H2):
Y = 1/Z_in + 1/Z_in_2 + 1/Z_in_3
Z_add = 1/Y
#Y_adm.append(Y)
#Z_adm.append(Z_add)
#Admittanz(Z_in, Z_in_2)
print('admittanz =', Y_adm)

    


def return_loss (Z_load, characteric_impedanz):
    S_11_of_f = np.abs((Z_load- characteric_impedanz)/ (Z_load + characteric_impedanz))
    S_11_dB = 20*np.log10(S_11_of_f)
    return S_11_of_f, S_11_dB
#S_11_in, S_11_in_dB = return_loss(Z_in, 50)
S_11_in, S_11_in_dB = return_loss(Z_add, 50)
S_11_load, S_11_load_dB = return_loss(Z_Load, 50)
S_11_load_2, S_11_load_dB_2 = return_loss(Z_Load_2, 50)
S_11_load_3, S_11_load_dB_3 = return_loss(Z_Load_3, 50)

# maximum S11_load 
ymax_load_dB = np.max(np.abs(S_11_load_dB))

print('ymax_load_dB=',ymax_load_dB)

# maximum S11_dB 
ymax_dB = np.max(np.abs(S_11_in_dB))

print('ymax_dB=',ymax_dB)






        


w = np.linspace(-4,-4,401) 
d = np.linspace(-4,-4,401)
     
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


ax1.plot(x,Z_Load.real)
ax1.plot(x,Z_Load_2.real)
ax1.plot(x,Z_Load_3.real)

ax1.plot(x,Z_Load.imag)
ax1.plot(x,Z_Load_2.imag)
ax1.plot(x,Z_Load_3.imag)

ax1.set_xlabel('Frequency in [Hz]', fontsize=6)
ax1.set_ylabel('IMPEDANCE [Ohm]', fontsize=6)
ax1.set_title('LOAD IMPEDANCE VS FREQUENCY', fontsize=8)
ax1.legend(['Real part', 'imaginary part','Real part_2', 'imaginary part_2'], fontsize=6)
ax1.annotate('Ymax Ohm',xy=(xmax, ymax), xytext=(xmax,ymax),fontsize=10)
#ax1.text(50e+9, 50, r'res. freq.:272.4 GHz , Imp 573 Ohm', fontsize=10)
#ax1.text(1.5e+10, 15, r'Max Imp: 9.37 Ohm', fontsize=10)

ax2.plot(x,S_11_load_dB)
ax2.plot(x,S_11_load_dB_2)
ax2.plot(x,S_11_load_dB_3)

ax2.set_xlabel('Frequency in [Hz]', fontsize=6)
ax2.set_ylabel('S11_Load [dB]', fontsize=6)
ax2.set_title('S11_Load in dB VS FREQUENCY', fontsize=8)
ax2.legend(['S_11_load_dB', 'S_11_load_dB_2'], fontsize=6)
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
    #print('BW_L1 =', BW_L1)
    #print('BW_L2 =', BW_L2)
    #print('BH_L1 =', BH_L1)
    #print('BH_L2 =', BH_L2)
    ax2.plot(*LineString(intersection).xy, 'o')
    #print('Bandwidth_load=',*LineString(intersection).xy)
    #ax2.annotate('f1 =283.7 GHz ',xy=(BW_L1, BH_L1-0.8),fontsize=10)
    #ax2.annotate('f2 =294.69 GHz',xy=(BW_L2, BH_L2),fontsize=10)
   
    
elif intersection.geom_type == 'Point':
    ax2.plot(*intersection.xy,x, 'o')
    #print((*intersection.xy,x).coords)


ax3.plot(x,Z_in.real)
ax3.plot(x,Z_in_2.real)
ax3.plot(x,Z_in_3.real)

ax3.plot(x,Z_in.imag)
ax3.plot(x,Z_in_2.imag)
ax3.plot(x,Z_in_3.imag)

ax3.set_title('INPUT IMPEDANCE vs FREQUENCY', fontsize=8)
ax3.set_xlabel('FREQUENCY in [Hz]', fontsize=6)
ax3.set_ylabel('INPUT IMPEDANCE [Ohm]', fontsize=6)
ax3.legend(['Real part', 'imaginary part'], fontsize=6)

ax4.plot(x,S_11_in_dB)

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