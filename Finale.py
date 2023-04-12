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


df = pd.read_csv('Z1_13.in', sep=' ',header=None)


"""print(df)"""
Frequences = df[0]
Real_part = df[1]


freq = (Frequences[750], Frequences[850])
rea  = (Real_part[750], Real_part[850])

x=[]    #new array frequence 
y=[]    #new real part 
C = speed_of_light


for i in range (750, 850):
    x.append(Frequences[i])
    y.append(Real_part[i])


#plt.plot(x, y)
plt.xlabel('Frequencies in [Hz]', fontsize=14)
plt.ylabel('real part in [Ohm]', fontsize=14)
plt.title('Optimization 1 real part vs frequnecies', fontsize=14)


# maximum 
ymax = np.max(y)
xpos = np.where(y == ymax)
xmax = np.array(x)[xpos]
print(xmax,ymax)

#plt.show()

Z_output = [] # content list of Z_input (Z_in real and Z_in imag)
Z_output_real = []   # list of real part 
Z_output_imag = []# list of imaginary part
Lamdas=[]
abss = []
Z_output_1 =[]   # Content list of Z_in 
Z_output_2 =[]  # content list of S_11
Z_dB = []

def Quarter_Wave(Z_l):
    #Z_output = [] # content list of Z_input
    #Z_0 = 70
    
    beta_quarter = []
    Z_opt = 70 
    C_impedanz = 50
    C = speed_of_light
    for f in (x):
        lamda = C/f 
        ###print(lamda)
        Lamdas.append(lamda)
        
    
        
        Beta_quarter = (2*np.pi)*xmax / C   # calcul of Beta
        length_wave = C/f*4             # length of Stripline
        Beta_quarter_l = Beta_quarter*length_wave   # B*l
        
        beta_quarter.append(Beta_quarter_l)
        Angle = np.tan(Beta_quarter_l)
        #print(Angle)


        Z_0 = np.sqrt(Z_opt*Z_l)                        # caracteristic impedance 
        Zin_num = Z_0*(Z_l + (1j*Z_0*np.tan(Beta_quarter_l)))      
        Zin_denom = (Z_0 + (1j*Z_l*np.tan(Beta_quarter_l)))  
        Z_in = Zin_num/Zin_denom                        # Input impedance 
        # check print(Z_in)
       
        Z_input = (Z_in.real, Z_in.imag)
        ###print(Z_in.real,Z_in.imag)
        

        Z_output_1.append(np.abs(Z_in))                           # list of input impedance 
        Z_output.append(Z_input)
        Z_output_real.append(Z_in.real)
        Z_output_imag.append(Z_in.imag)
        valeur_absolue = np.abs(Z_output)
        abss.append(valeur_absolue)
        
        S_11_of_f = np.abs((np.abs(Z_in)- C_impedanz)/ (np.abs(Z_in) + C_impedanz))
        #print(S_11_of_f)
        S_11_dB = 20*np.log10(S_11_of_f)
        Z_output_2.append(S_11_of_f) 
        
        Z_dB.append(S_11_dB)
        
        #print(Z_dB)
        #print(type(Z_dB))
        #print(np.shape(x))
        
       
        
        #plt.plot(x, Z_dB)
        #plt.show()
        
        
    ##return Z_output
new_list = Z_output
###print(Quarter_Wave(ymax))
Quarter_Wave(ymax)

#plt.plot(x,np.abs(Z_output))
plt.plot(x,(Z_dB))
plt.show()




#new_list = 
df1 = pd.DataFrame(Z_output_real)
df2 = pd.DataFrame(Z_output_imag)
df3 = pd.DataFrame(x)
df4 = pd.DataFrame(Z_output_2)
df5 = pd.DataFrame(Z_output_2)
writer = pd.ExcelWriter('VALUE.xlsx', engine='xlsxwriter')
df1.to_excel(writer, sheet_name='real', index=False)
df2.to_excel(writer, sheet_name='imag', index=False)
df3.to_excel(writer, sheet_name='frequence', index=False)
df4.to_excel(writer, sheet_name='Z_output_2', index=False)
df4.to_excel(writer, sheet_name='Z_output_2', index=False)
writer.save()
