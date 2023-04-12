# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:32:22 2023

@author: Moussa Sarr Ndiaye
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
data = pd.read_excel("Z vs inset.xlsx", sheet_name=None)

# Loop through each sheet (optimization)
fig, ax = plt.subplots()
for sheet_name, sheet_data in data.items():

    # Extract the relevant columns
    Length_Inset = sheet_data.iloc[:, 2]
    Impedance = sheet_data.iloc[:, 0]
    reflectivity = sheet_data.iloc[:, 1]

    # Create the plots
    
    plt.plot(Length_Inset, Impedance)
    plt.scatter(Length_Inset, Impedance)
    #plt.set_xlabel("Frequency")
    plt.xlabel("Length_Inset")
    #plt.set_ylabel("Real Part")
    plt.ylabel("Impedance")
    #plt.set_title(f"{sheet_name} Optimization")
    
    #deco
    #plt.spines['bottom'].set_linewidth(2)
    #plt.spines['left'].set_linewidth(2)
    #plt.spines['top'].set_visible(False)
    #plt.spines['right'].set_visible(False)
    
    #plt.xaxis.set_ticks_position('bottom')
    #plt.yaxis.set_ticks_position('left')
    
    
    plt.tick_params(labelsize=14)
    
    #plt.yaxis.grid(True, which='major')
    #plt.xaxis.grid(True, which='major')
    #deco


    
    plt.grid()
    plt.show()


    #ax2.plot(freq, reflectivity)
    #ax2.set_xlabel("Frequency")
    #ax2.set_ylabel("Reflectivity")

    # Save the plot
    #plt.savefig(f"{sheet_name}_plot.png")