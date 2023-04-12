# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:18:56 2022

@author: Moussa Sarr Ndiaye
"""
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
import control 

# Low-pass Function
wl = 1 #rad/s
Tfl = 1/wl
num = np.array([1])
den = np.array([Tfl , 1])
HL = control.tf(num, den)
# High-pass Function
wh = 10 #rad/s
Tfh = 1/wh
num = np.array([Tfh, 0])
den = np.array([Tfh, 1])
HH = control.tf(num, den)
# Band-stop Function
HBS = control.parallel(HL, HH)
# Frequencies
w_start = 0.01
w_stop = 1000
step = 0.01
N = int ((w_stop-w_start )/step) + 1
w = np.linspace (w_start , w_stop , N)
# Frequency Response Plot
mag , phase_rad , w = control.bode_plot(HBS, w, dB=True,
Plot=False)
# Convert to dB
mag_db = 20 * np.log10(mag)
plt.figure()
plt.semilogx(w, mag_db)
plt.title("Bandpass Filter")
plt.grid(b=None, which='major', axis='both')
plt.grid(b=None, which='minor', axis='both')
plt.ylabel("Magnitude (dB)")
#plt.show()