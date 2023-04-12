# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 23:02:37 2022

@author: Moussa Sarr Ndiaye
"""





import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

x = np.arange(0, 1000)
f = np.arange(0, 1000)
g = np.sin(np.arange(0, 10, 0.01) * 2) * 1000

plt.plot(x, f)
plt.plot(x, g)

first_line = LineString(np.column_stack((x, f)))
second_line = LineString(np.column_stack((x, g)))
intersection = first_line.intersection(second_line)

if intersection.geom_type == 'MultiPoint':
    plt.plot(*LineString(intersection).xy, 'o')
elif intersection.geom_type == 'Point':
    plt.plot(*intersection.xy, 'o')