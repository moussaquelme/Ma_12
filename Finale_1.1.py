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
import control 
from shapely.geometry import LineString


df = pd.read_csv('Z1_13.in', sep=' ',header=None)
#print(df)

"""print(df)"""
Frequences = df[0]
Real_part = df[1]
Imag_part = df[2]


freq = (Frequences[100], Frequences[1000])
rea  = (Real_part[100], Real_part[1000])
img = (Imag_part[100], Imag_part[1000])

x=[]    #new array frequence 
y=[]    #new real part 
z=[]    #new imaginary part 

C = speed_of_light
#print(Frequences[100],Frequences[1000])


for i in range (1, 1000):
    x.append(Frequences[i])
    y.append(Real_part[i])
    z.append(Imag_part[i])


#plt.plot(x, y)
#plt.xlabel('Frequencies in [Hz]', fontsize=14)
#plt.ylabel('real part in [Ohm]', fontsize=14)
#plt.title('Optimization 1 real part vs frequnecies', fontsize=14)


# maximum 
ymax = np.max(y)
xpos = np.where(y == ymax)
xmax = np.array(x)[xpos]
print(xmax,ymax)


plt.show()

Z_output = [] # content list of Z_input (Z_in real and Z_in imag)
Z_output_real = []   # list of real part 
Z_output_imag = []# list of imaginary part
Lamdas=[]
abss = []
Z_output_1 =[]   # Content list of Z_in 
Z_output_2 =[]  # content list of S_11
Z_dB = []
beta_quarter = []

def Quarter_Wave(Z_l):
    #Z_output = [] # content list of Z_input
    #Z_0 = 70
    
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
        lamda_f = C/f 
        ###print(lamda_f)
        Lamdas.append(lamda_f)
        Beta = (2*np.pi)/lamda_f   # calcul of Beta
        Beta_quarter_l = Beta * length_wave   # B*l
        beta_quarter.append(Beta_quarter_l)
        Angle = np.tan(Beta_quarter_l)
        #print(Angle)

        Zin_num = Z_0*(Z_l + 1j*Z_0*np.tan(Beta_quarter_l))      
        Zin_denom = (Z_0 + 1j*Z_l*np.tan(Beta_quarter_l))  
        
        Z_in = Zin_num/Zin_denom                        # Input impedance 
        ##print(Z_in)
       
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
        
        
    
new_list = Z_output
Quarter_Wave(ymax)

#fig, axs =plt.subplots(2,2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

#axs.figure(figsize=(10,10))
ax1.plot(x,Z_dB)
ax1.set_title('Return Loss over frequency')
ax1.set_xlabel('Frequency in [Hz]')
ax1.set_ylabel('Return loss in [dB]')



ax2.plot(x,Z_output_real)
ax2.set_title('Zin_real vs frequency')
ax2.set_xlabel('Frequency in [Hz]')
ax2.set_ylabel('Zin_Impedanz real [Ohm]')


ax3.plot(x,Z_output_imag ,'r')
ax3.set_title('Z_in imag vs frequency')
ax3.set_xlabel('Frequency in [Hz]')
ax3.set_ylabel('Zin_Impedanz imaginary in [Ohm]')

ax4.plot(x,y ,'r')
#ax5=ax4.twinx()
ax4.plot(x,z)
ax4.set_title('Load impedanz vs frequency')
ax4.set_xlabel('Frequency in [Hz]')
ax4.set_ylabel('REAL PART LOAD IMPEDANCE [Ohm]')

plt.tight_layout()
ax1.grid(b=None, which='major', axis='both')
ax2.grid(b=None, which='major', axis='both')
ax3.grid(b=None, which='major', axis='both')
ax4.grid(b=None, which='major', axis='both')
#plt.show()
#fig.savefig('opt3.png')






#new_list = 
df1 = pd.DataFrame(Z_output_real)
df2 = pd.DataFrame(Z_output_imag)
df3 = pd.DataFrame(x)
df4 = pd.DataFrame(Z_output_2)
df5 = pd.DataFrame(Z_output)
df6 = pd.DataFrame(Z_dB)
df7 = pd.DataFrame(Lamdas)
writer = pd.ExcelWriter('VALUE.xlsx', engine='xlsxwriter')
df1.to_excel(writer, sheet_name='real', index=False)
df2.to_excel(writer, sheet_name='imag', index=False)
df3.to_excel(writer, sheet_name='frequence', index=False)
df4.to_excel(writer, sheet_name='Z_output_2', index=False)
df5.to_excel(writer, sheet_name='Z_output', index=False)
df6.to_excel(writer, sheet_name='S11_dB', index=False)
d = df6.to_excel(writer, sheet_name='S11_dB', index=False)
df7.to_excel(writer, sheet_name='Lamdas', index=False)
writer.save()



