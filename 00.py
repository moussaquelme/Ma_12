# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:40:43 2023

@author: Moussa Sarr Ndiaye
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmath import rect
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class AntennaSystem:
    def __init__(self, file_310, file_270):
        self.df_310 = pd.read_csv(file_310)
        self.df_270 = pd.read_csv(file_270)
        self.Frequencies = np.array(self.df_310['Frequency (MHz)'])
        self.Z_Load = self.create_complex_array(self.df_310['LoadMag'], self.df_310['LoadPhase'])
        self.Z_Load_2 = self.create_complex_array(self.df_270['LoadMag'], self.df_270['LoadPhase'])
        self.Z_in = None
        self.Z_in_2 = None
        self.admittance = None
        
    @staticmethod
    def create_complex_array(mag, phase):
        return np.array([rect(m, np.deg2rad(p)) for m, p in zip(mag, phase)])
    
    def quarter_wave(self):
        if self.Frequencies is None:
            raise ValueError("Frequencies have not been defined")
        self.Z_in = np.zeros((len(self.Frequencies), 1), np.complex64)
        self.Z_in_2 = np.zeros((len(self.Frequencies), 1), np.complex64)
        for i, f in enumerate(self.Frequencies):
            l = 75 / (4 * f)
            self.Z_in[i] = 75 * ((self.Z_Load[i] / 75) ** 2 + 2 * (1j * np.tan(2 * np.pi * l * f)) / 75) / \
                            (self.Z_Load[i] / 75 + 2 * (1j * np.tan(2 * np.pi * l * f)) / 75)
            self.Z_in_2[i] = 75 * ((self.Z_Load_2[i] / 75) ** 2 + 2 * (1j * np.tan(2 * np.pi * l * f)) / 75) / \
                              (self.Z_Load_2[i] / 75 + 2 * (1j * np.tan(2 * np.pi * l * f)) / 75)
    
    def admittance(self):
        if self.Z_in is None or self.Z_in_2 is None:
            raise ValueError("Z_in or Z_in_2 have not been defined")
        Y_in = 1 / self.Z_in
        Y_in_2 = 1 / self.Z_in_2
        self.admittance = Y_in + Y_in_2
    
    @staticmethod
    def return_loss(Z_in):
        Z_L = 75
        Z_0 = 50
        return 20 * np.log10(abs((Z_in - Z_L) / (Z_in + Z_L)))
    
    def plot_impedance(self):
        if self.Frequencies is None or self.Z_Load is None or self.Z_in is None or self.admittance is None:
            raise ValueError("Frequencies, Z_Load, Z_in, or admittance have not been defined")
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        fig.suptitle('Impedance and Admittance of Antenna System', fontsize=14)
        
        # Plot load impedance
        ax[0, 0].plot(self.Z_Load.real, self.Z_Load.imag, 'b', label='Load')
        
        
        def plot_results(self):
            # Create a 2x2 grid of subplots
            fig, ax = plt.subplots(2, 2, figsize=(12, 10))
            
            # Set plot titles
            ax[0, 0].set_title('Load Impedance')
            ax[0, 1].set_title('Input Impedance')
            ax[1, 0].set_title('Return Loss')
            ax[1, 1].set_title('Intersection Points')
            
            # Plot load impedance
            ax[0, 0].plot(self.Z_Load.real, self.Z_Load.imag, 'b', label='Load')
            ax[0, 0].plot(self.Z_Load_2.real, self.Z_Load_2.imag, 'b')
            ax[0, 0].set_xlabel('Real')
            ax[0, 0].set_ylabel('Imaginary')
            ax[0, 0].legend()
            
            # Plot input impedance
            ax[0, 1].plot(self.Z_in.real, self.Z_in.imag, 'r', label='Input')
            ax[0, 1].plot(self.Z_in_2.real, self.Z_in_2.imag, 'r')
            ax[0, 1].set_xlabel('Real')
            ax[0, 1].set_ylabel('Imaginary')
            ax[0, 1].legend()
            
            # Plot return loss
            ax[1, 0].plot(self.Frequencies, self.return_loss(self.Z_in), 'g', label='Return Loss')
            ax[1, 0].set_xlabel('Frequency (MHz)')
            ax[1, 0].set_ylabel('Return Loss (dB)')
            ax[1, 0].legend()
            
            # Plot intersection points
            ax[1, 1].plot(self.Frequencies, self.Z_in.real, 'r', label='Real')
            ax[1, 1].plot(self.Frequencies, self.Z_in.imag, 'b', label='Imaginary')
            ax[1, 1].set_xlabel('Frequency (MHz)')
            ax[1, 1].set_ylabel('Impedance')
            ax[1, 1].legend()
            
            # Show the plot
            plt.show()