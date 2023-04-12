# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 20:58:15 2023

@author: Moussa Sarr Ndiaye
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV files and concatenate them into one dataframe
#df1 = pd.read_csv('Z ParameterRI_100um.csv', header=None, names=['InsetL [um]', 'ResFreq [GHz]', 'Freq [GHz]', 're(Z(1,1)) []', 'im(Z(1,1)) []'])
df1 = pd.read_csv('Z ParameterRI_100um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
df2 = pd.read_csv('Z ParameterRI_80um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
df3 = pd.read_csv('Z ParameterRI_60um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
df4 = pd.read_csv('Z ParameterRI_15um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
df5 = pd.read_csv('Z ParameterRI_25um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
df6 = pd.read_csv('Z ParameterRI_35um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
df7 = pd.read_csv('Z ParameterRI_45um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
df8 = pd.read_csv('Z ParameterRI_5um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])

# Create a new dataframe with just the real part and frequency columns
df_real = df[['frequency', 'real']]

# Plot the data
plt.plot(df_real['frequency'], df_real['real'])
plt.xlabel('Frequency')
# Set x-axis limits
plt.xlim(260, 330)
plt.ylabel('Real Part')
plt.title('Real Part vs Frequency')
#plt.legend('100um', '80um', '60um', '15um', '25um', '35um', '45um', '5um')
plt.legend()
plt.tight_layout()

plt.grid(b=None, which='major', axis='both')
plt.show()