# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 23:17:06 2023

@author: Moussa Sarr Ndiaye
"""

import numpy as np
import pandas as pd
import math
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt
from shapely.geometry import LineString


def load_csv_data(file1, file2):
    df1 = pd.read_csv(file1, sep=',', header=None)
    df2 = pd.read_csv(file2, sep=',', header=None)

    frequencies = df1[0]
    magnitude1 = df1[1]
    phase1 = df1[2]
    magnitude2 = df2[1]
    phase2 = df2[2]

    return frequencies, magnitude1, phase1, magnitude2, phase2


def calculate_complex_impedance(frequencies, magnitude, phase):
    complex_impedance = np.zeros(len(frequencies), np.complex64)

    for i in range(len(frequencies)):
        complex_impedance[i] = magnitude[i] * np.exp(1j * math.radians(phase[i]))
        print(complex_impedance[i])
    return complex_impedance
    

def calculate_input_impedance(Z_l, Z_l_2, x):
    Z_opt = 70
    C_impedanz = 50
    Z_0 = np.sqrt(Z_opt * np.max(Z_l.real))
    Z_0_2 = np.sqrt(Z_opt * np.max(Z_l_2.real))
    lamda = speed_of_light / np.array(x)[np.where(Z_l.real == np.max(Z_l.real))[0]]
    lamda_2 = speed_of_light / np.array(x)[np.where(Z_l_2.real == np.max(Z_l_2.real))[0]]

    length_wave = lamda / 4 * 80 / 90
    length_wave_2 = lamda_2 / 4 * 80 / 90

    Z_in = np.zeros(len(x), np.complex64)
    Z_in_2 = np.zeros(len(x), np.complex64)

    for idx, f in enumerate(x):
        Lamda_f = speed_of_light / f
        Lamda_f_2 = speed_of_light / f

        Beta_quarter_l = ((2 * np.pi) / Lamda_f) * length_wave
        Beta_quarter_l_2 = ((2 * np.pi) / Lamda_f_2) * length_wave_2

        Zin_num = Z_0 * (Z_l[idx] + 1j * Z_0 * np.tan(Beta_quarter_l))
        Zin_num_2 = Z_0_2 * (Z_l_2[idx] + 1j * Z_0_2 * np.tan(Beta_quarter_l_2))

        Zin_denom = (Z_0 + 1j * Z_l[idx] * np.tan(Beta_quarter_l))
        Zin_denom_2 = (Z_0_2 + 1j * Z_l_2[idx] * np.tan(Beta_quarter_l_2))

        Z_in[idx] = Zin_num / Zin_denom
        Z_in_2[idx] = Zin_num_2 / Zin_denom_2

    return Z_in, Z_in_2


def calculate_admittance(Z_in, Z_in_2):
    Y = 1 / Z_in + 1 / Z_in_2
    Z_add = 1 / Y

    return Z_add


def calculate_return_loss(Z_load, characteristic_impedance):
    S_11_of_f = np.abs((Z_load - characteristic_impedance) / (Z_load + characteristic_impedance))
    return_loss = -20 * np.log10(S_11_of_f)
    return return_loss, S_11_of_f

def plot_all_graphs(frequencies, Z_l, Z_l_2, Z_in, Z_in_2, Z_add, return_loss, S_11_of_f):
    fig, axs = plt.subplots(4, 1, figsize=(8, 15))

    # Plot Z_l and Z_l_2
    axs[0].plot(frequencies, Z_l.real, label='Z_l Real')
    axs[0].plot(frequencies, Z_l_2.real, label='Z_l_2 Real')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Real Impedance')
    axs[0].set_title('Z_l and Z_l_2 Real Parts vs Frequency')
    axs[0].legend()
    axs[0].grid()

    # Plot Z_in and Z_in_2
    axs[1].plot(frequencies, Z_in.real, label='Z_in Real')
    axs[1].plot(frequencies, Z_in_2.real, label='Z_in_2 Real')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Real Impedance')
    axs[1].set_title('Z_in and Z_in_2 Real Parts vs Frequency')
    axs[1].legend()
    axs[1].grid()

    # Plot Z_add
    axs[2].plot(frequencies, Z_add.real, label='Z_add Real')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Real Impedance')
    axs[2].set_title('Z_add Real Part vs Frequency')
    axs[2].legend()
    axs[2].grid()

    # Plot return_loss and S_11_of_f
    axs[3].plot(frequencies, return_loss, label='Return Loss')
    axs[3].plot(frequencies, -20 * np.log10(S_11_of_f), label='S_11')
    axs[3].set_xlabel('Frequency (Hz)')
    axs[3].set_ylabel('Return Loss (dB)')
    axs[3].set_title('Return Loss and S_11 vs Frequency')
    axs[3].legend()
    axs[3].grid()

    plt.tight_layout()
    plt.show()

def main():
    file1 = "310.csv"
    file2 = "270.csv"
    frequencies, magnitude1, phase1, magnitude2, phase2 = load_csv_data(file1, file2)

    Z_l = calculate_complex_impedance(frequencies, magnitude1, phase1)
    Z_l_2 = calculate_complex_impedance(frequencies, magnitude2, phase2)
    
    Z_in, Z_in_2 = calculate_input_impedance(Z_l, Z_l_2, frequencies)
    
    Z_add = calculate_admittance(Z_in, Z_in_2)
    
    characteristic_impedance = 50
    
    return_loss, S_11_of_f = calculate_return_loss(Z_add, characteristic_impedance)
    
    plot_all_graphs(frequencies, Z_l, Z_l_2, Z_in, Z_in_2, Z_add, return_loss, S_11_of_f)

if __name__ == "__main__":
    main()