from cmath import pi, sqrt
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.constants import speed_of_light
import os

from functools import partial
from multiprocessing import Pool, TimeoutError

#from numba import jit
import time
import csv 
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar
from array import *
from scipy.constants import speed_of_light


df = pd.read_csv('Z1_13.in', sep=' ',header=None)


"""print(df)"""
Frequences = df[0]
Real_part = df[1]
Imag_part = df[2]

freq = (Frequences[100], Frequences[1000])
rea  = (Real_part[100], Real_part[1000])

x=[]    #new array frequence 
y=[]    #new real part 
z=[]    #new imaginary part 
C = speed_of_light


for i in range (1, 1000):
    x.append(Frequences[i])
    y.append(Real_part[i])
    z.append(Imag_part[i])


plt.plot(x, y)
plt.plot(x,z)
plt.xlabel('Frequencies in [Hz]', fontsize=14)
plt.ylabel('real part in [Ohm]', fontsize=14)
plt.title('Optimization 13 real part vs frequnecies', fontsize=14)


# maximum 
ymax = np.max(y)
xpos = np.where(y == ymax)
xmax = np.array(x)[xpos]
print(xmax,ymax)

plt.show()

Z_output = [] # content list of Z_input
Z_output_real = []
Z_output_imag = []
Lamdas=[]


def Quarter_Wave(Z_l):
    #Z_output = [] # content list of Z_input
    #Z_0 = 70
    
    beta_quarter = []
    Z_opt = 70 
    C = speed_of_light
    Z_0 = np.sqrt(Z_opt* ymax)                        # caracteristic impedance  

    lamda = C/xmax        
    Beta_quarter = (2*np.pi)/ lamda   # calcul of Beta
    length_wave = lamda/4             # length of Stripline bei resonanz 
    
    
    for f in (x):
        lamda_f = C/f 
        print(lamda_f)
        Lamdas.append(lamda_f)
        
    
        """Beta_quarter = (2*np.pi)/ lamda   # calcul of Beta
        length_wave = lamda/4             # length of Stripline"""

        Beta = (2*np.pi)/ lamda_f
        Beta_quarter_l = Beta *length_wave   # B*l
        #print(Beta_quarter_l)
        beta_quarter.append(Beta_quarter_l)


        #Z_0 = np.sqrt(Z_opt*Z_l)                        # caracteristic impedance 
        Zin_num = Z_0*(Z_l + 1j*Z_0*np.tan(Beta_quarter_l))      
        Zin_denom = (Z_0 + 1j*Z_l*np.tan(Beta_quarter_l))          
        Z_in = Zin_num/Zin_denom                        # Input impedance 
        print(Z_in)
        Z_input = (Z_in.real, Z_in.imag)

        #Z_output.append(Z_in)                           # list of input impedance 
        Z_output.append(Z_input)
        Z_output_real.append(Z_in.real)
        Z_output_imag.append(Z_in.imag)
        
        

    ##return Z_output
new_list = Z_output
print(Quarter_Wave(ymax))

#new_list = 
df1 = pd.DataFrame(Z_output_real)
df2 = pd.DataFrame(Z_output_imag)
df3 = pd.DataFrame(x)
df4 = pd.DataFrame(Lamdas)
writer = pd.ExcelWriter('VALUE.xlsx', engine='xlsxwriter')
df1.to_excel(writer, sheet_name='real', index=False)
df2.to_excel(writer, sheet_name='imag', index=False)
df3.to_excel(writer, sheet_name='frequence', index=False)
df4.to_excel(writer, sheet_name='lamdas', index=False)
writer.save()
