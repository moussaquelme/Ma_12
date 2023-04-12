# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 23:34:32 2023

@author: Moussa Sarr Ndiaye
"""

import pandas as pd
import numpy as np
import math
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt
from shapely.geometry import LineString

class PatchAntenna:
    def __init__(self, impedance_file, reflection_file):
        self.df_impedance = pd.read_csv(impedance_file, sep=',', header=None)
        self.df_reflection = pd.read_csv(reflection_file, sep=',', header=None)
        self.freqs = self.df_impedance[1]
        self.mag_impedance = self.df_impedance[2]
        self.phase_impedance = self.df_impedance[3]
        self.mag_reflection = self.df_reflection[2]
        self.phase_reflection = self.df_reflection[3]
        self.speed_of_light = speed_of_light
    
    def calculate_input_impedance(self):
        z_load = self.mag_impedance * np.exp(1j * np.radians(self.phase_impedance))
        z_reflection = self.mag_reflection * np.exp(1j * np.radians(self.phase_reflection))
        z_0 = np.sqrt(70 * np.max(self.mag_impedance)) # characteristic impedance
        lamda = self.speed_of_light / self.freqs[np.argmax(self.mag_impedance)] # wavelength at resonance
        beta_quarter = 2 * np.pi / lamda # beta at quarter wavelength
        length_wave = lamda / 4 * 80 / 90 # length of stripline at resonance
        z_in = np.zeros((len(self.freqs), 1), np.complex64)
        
        for idx, freq in enumerate(self.freqs):
            lamda_f = self.speed_of_light / freq
            beta_quarter_l = ((2 * np.pi) / lamda_f) * length_wave
            z_in_num = z_0 * (z_load[idx] + 1j * z_0 * np.tan(beta_quarter_l))
            z_in_denom = (z_0 + 1j * z_load[idx] * np.tan(beta_quarter_l))
            z_in[idx] = z_in_num / z_in_denom # input impedance
        return z_in
    
    def calculate_S11(self, z_in):
        s11 = np.abs((z_in - 50) / (z_in + 50))
        s11_db = 20 * np.log10(s11)
        return s11_db
    
    def plot_impedance(self):
        plt.plot(self.freqs, self.mag_impedance, label='Magnitude')
        plt.plot(self.freqs, self.phase_impedance, label='Phase')
        plt.title('Impedance vs Frequency')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Impedance [Ohm]')
        plt.legend()
        plt.show()
    
    def plot_reflection(self):
        plt.plot(self.freqs, self.mag_reflection, label='Magnitude')
        plt.plot(self.freqs, self.phase_reflection, label='Phase')
        plt.title('Reflection vs Frequency')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Reflection Coefficient')
        plt.legend()
        plt.show()
        
    def plot_input_impedance(self, z_in):
        plt.plot(self.freqs, z_in.real, label='Real part')
        plt.plot(self.freqs, z_in.imag, label='Imaginary part')
        plt.title('Input Impedance vs Frequency')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Input Impedance [Ohm]')
        plt.legend()
        
