# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:11:36 2023

@author: Moussa Sarr Ndiaye
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath
from scipy.constants import speed_of_light
import os

class AntennaSystem:
    def __init__(self):
        self.df_310 = pd.read_csv('310.csv')
        self.df_270 = pd.read_csv('270.csv')
        self.Frequencies = np.array(self.df_310['Frequency (MHz)'])
        self.Z_Load = self.create_complex_array(self.df_310['LoadMag'], self.df_310['LoadPhase'])
        self.Z_Load_2 = self.create_complex_array(self.df_270['LoadMag'], self.df_270['LoadPhase'])
        self.Z_in = np.zeros((len(self.Frequencies), 1), np.complex64)
        self.Z_in_2 = np.zeros((len(self.Frequencies), 1), np.complex64)
        self.Y = np.zeros((len(self.Frequencies), 1), np.complex64)
        self.VSWR = np.zeros((len(self.Frequencies), 1), np.float32)
        
    def create_complex_array(self, magnitude, phase):
        real = magnitude * np.cos(np.deg2rad(phase))
        imag = magnitude * np.sin(np.deg2rad(phase))
        return real + 1j * imag


    def extract_data(self):
        Frequencies = self.df[1]
        magnitude = self.df[2]
        phase = self.df[3]
        mag = self.hf[2]
        phi = self.hf[3]

        self.x = []  # new array frequency
        self.y = []  # new real part
        self.z = []  # new imaginary part
        self.u = []
        self.v = []

        for i in range(0, 401):
            self.x.append(Frequencies[i])
            self.y.append(magnitude[i])
            self.z.append(phase[i])
            self.u.append(mag[i])
            self.v.append(phi[i])

        self.Z_Load = np.zeros((len(self.x), 1), np.complex64)
        self.Z_Load_2 = np.zeros((len(self.x), 1), np.complex64)

        for i in range(len(self.x)):
            self.Z_Load[i] = self.y[i] * np.exp(1j * np.radians(self.z[i]))
            self.Z_Load_2[i] = self.u[i] * np.exp(1j * np.radians(self.v[i]))

    def quarter_wave(self):
        if self.Frequencies is None:
            raise ValueError("Frequencies have not been defined")
            self.Z_in = np.zeros((len(self.Frequencies), 1), np.complex64)
            self.Z_in = np.zeros((len(self.Frequencies), 1), np.complex64)
            self.Z_in_2 = np.zeros((len(self.Frequencies), 1), np.complex64)

        for i in range(len(self.Frequences)):
            wavelength = 3e8 / (self.Frequences[i] * 1e6)
            z0 = np.sqrt(self.Z_Load[i] * self.Z_Load_2[i])
            beta = 2 * np.pi / wavelength
            d = wavelength * 0.25
            angle = beta * d
            self.Z_in[i] = z0 * (self.Z_Load[i] + 1j * z0 * np.tan(angle)) / (z0 + 1j * self.Z_Load[i] * np.tan(angle))
            self.Z_in_2[i] = z0 * (self.Z_Load_2[i] + 1j * z0 * np.tan(angle)) / (z0 + 1j * self.Z_Load_2[i] * np.tan(angle))

        return self.Z_in, self.Z_in_2

    def admittance(self):
        Z_in, Z_in_2 = self.quarter_wave()
        Y = 1 / Z_in + 1 / Z_in_2
        self.Z_add = 1 / Y

    def return_loss(self, Z_load, characteristic_impedance):
        S_11_of_f = np.abs((Z_load - characteristic_impedance) / (Z_load + characteristic_impedance))
        S_11_dB = 20 * np.log10(S_11_of_f)
        return S_11_of_f, S_11_dB

    def calculate_return_loss(self):
        self.admittance()
        self.S_11_in, self.S_11_in_dB = self.return_loss(self.Z_add, 50)
        self.S_11_load, self.S_11_load_dB = self.return_loss(self.Z_Load, 50)
        self.S_11_load_2, self.S_11_load_dB_2 = self.return_loss(self.Z_Load_2, 50)

    def plot(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        # Add the plotting commands as shown in your code

        plt.tight_layout()
        ax1.grid(b=None, which='major', axis='both')
        ax2.grid(b=None, which='major', axis='both')
        ax3.grid(b=None, which='major', axis='both')
        ax4.grid(b=None, which='major', axis='both')
        plt.show()
        
        
if __name__ == "__main__":
    antenna_system = AntennaSystem('310.csv', '270.csv')
    antenna_system.calculate_return_loss()
    antenna_system.plot()
    
