# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 19:50:59 2023

@author: Moussa Sarr Ndiaye
"""

import time
import csv 
import pandas as pd
# from scipy.optimize import curve_fit, minimize_scalar
from array import *
from scipy.constants import speed_of_light
# import control 
from shapely.geometry import LineString
import shapely.geometry
import shapely.wkt
from shapely.geometry import mapping, shape
# import control
import math




   


df = pd.read_csv('Zin_MagDeg_HFSS_40um_Inset.csv', sep=',',header=None)
#df = pd.read_csv('Zin_MagDeg_HFSS_40um_Inset.csv', sep=',')
#df = pd.read_csv('Z1_13.in', sep=' ',header=None)




print(df)

Frequences = df[0]
magnitude = df[1]
phase = df[2]

print(Frequences)