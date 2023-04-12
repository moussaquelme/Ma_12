# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 22:47:22 2023

@author: Moussa Sarr Ndiaye
"""

import numpy as np
import skrf as rf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skrf as rf



impedance = [124.2, 77.9, 51.8, 21.7, 0]
Inset = [15, 45, 60, 80, 100]







# Set figure size
plt.figure(figsize=(8, 6))

plt.plot(Inset, impedance)
plt.scatter(Inset, impedance)

# Set axis labels and title
plt.xlabel('Inset (um)')
plt.ylabel('Impedance R(Z)')
plt.title('Impedance R(Z) vs Inset length')
plt.grid()
# Set figure size
plt.figure(figsize=(8, 6))
plt.show()