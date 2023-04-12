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
for sheet_name, sheet_data in data.items():

    # Extract the relevant columns
    freq = sheet_data.iloc[:, 2]
    real_part = sheet_data.iloc[:, 0]
    reflectivity = sheet_data.iloc[:, 1]

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(freq, real_part)
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Real Part")
    ax1.set_title(f"{sheet_name} Optimization")

    #ax2.plot(freq, reflectivity)
    #ax2.set_xlabel("Frequency")
    #ax2.set_ylabel("Reflectivity")

    # Save the plot
    #plt.savefig(f"{sheet_name}_plot.png")