# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:52:26 2023

@author: Moussa Sarr Ndiaye
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file into a pandas dataframe
df = pd.read_excel('L_inset3.xlsx', sheet_name=None)

# Loop through each sheet in the dataframe and plot the real part vs frequency
fig, ax = plt.subplots()
for sheet_name, sheet_data in df.items():
    #freq = sheet_data['real']
    #real_part = sheet_data['frequency']
    
    freq = sheet_data.iloc[:, 1]
    real_part = sheet_data.iloc[:, 2]
    
    ax.plot(freq, real_part, label=sheet_name)

# Set the title and axis labels
#ax.set_title('Real Part vs Frequency')
ax.set_xlabel('Frequency in Hz')
ax.set_ylabel('Impedance {R[Z]}')


#deco
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


ax.tick_params(labelsize=14)

ax.yaxis.grid(True, which='major')
ax.xaxis.grid(True, which='major')
#deco


# Add a legend to the plot
ax.legend()

#plt.grid()
# Show the plot
plt.show()





