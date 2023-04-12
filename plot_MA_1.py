# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:49:36 2023

@author: Moussa Sarr Ndiaye
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV files and concatenate them into one dataframe
df1 = pd.read_csv('Z ParameterRI_100um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
df2 = pd.read_csv('Z ParameterRI_80um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
df3 = pd.read_csv('Z ParameterRI_60um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
df4 = pd.read_csv('Z ParameterRI_15um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
#df5 = pd.read_csv('Z ParameterRI_25um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
#df6 = pd.read_csv('Z ParameterRI_35um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
df7 = pd.read_csv('Z ParameterRI_45um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])
#df8 = pd.read_csv('Z ParameterRI_5um.csv', header=None, names=['length', 'resonanz_freq', 'frequency', 'real', 'imag'])

# Create separate dataframes for the real part and frequency data for each CSV file
df1_real = df1[['frequency', 'real']]
df2_real = df2[['frequency', 'real']]
df3_real = df3[['frequency', 'real']]
df4_real = df4[['frequency', 'real']]
#df5_real = df5[['frequency', 'real']]
#df6_real = df6[['frequency', 'real']]
df7_real = df7[['frequency', 'real']]
#df8_real = df8[['frequency', 'real']]

# Plot the data from each file separately
plt.plot(df1_real['frequency'], df1_real['real'], label='Inset_length: 100um')
plt.plot(df2_real['frequency'], df2_real['real'], label='Inset_length: 80um')
plt.plot(df3_real['frequency'], df3_real['real'], label='Inset_length: 60um')
plt.plot(df4_real['frequency'], df4_real['real'], label='Inset_length: 15um')
#plt.plot(df5_real['frequency'], df5_real['real'], label='Inset_length: 25um')
#plt.plot(df6_real['frequency'], df6_real['real'], label='Inset_length: 35um')
plt.plot(df7_real['frequency'], df7_real['real'], label='Inset_length: 45um')
#plt.plot(df8_real['frequency'], df8_real['real'], label='Inset_length: 5um')

# Add a legend for each file
plt.legend()
plt.grid()

# Set x-axis limits
plt.xlim(280, 320)
#plt.ylim(0, 140)

# Set axis labels and title
plt.xlabel('Frequency (Hz)')
plt.ylabel('Real Part')
plt.title('Real Part vs Frequency')

# Set figure size
plt.figure(figsize=(8, 6))

# Show the plot
#plt.grid()
plt.show()



