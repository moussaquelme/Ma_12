# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 12:12:58 2023

@author: Moussa Sarr Ndiaye
"""

from patch_util.patch import design_patch, input_impedance, inset_feed_position, \
    get_directivity, patch_eh_plane_plot, surface_plot, getGs


freq = 2.4e9
Er = 4.4
h = 1.6 * 10 ** -3
v = 3 * 10 ** 8

W, L = design_patch(Er, h, freq)


Rin = input_impedance(freq, W, L)
print('Inset Feed Position : ', inset_feed_position(Rin, L))

G1, G12 = getGs(freq, W, L)
print('G1 : ', G1)
print('G12 : ', G12)

I1 = 1.863
I2 = 3.59801

d1, d2 = get_directivity(G1, G12, W, freq, I1, I2)

print('Directivity : ', d1, ' dB')
print('Directivity : ', d2, ' dB')

fields = patch_eh_plane_plot(freq, W, L, h, Er)
surface_plot(fields)



import patch_antenna as pa

# resonant frequency in Hz
freq = 2.4 * 10 ** 9

# dielectric constant
er = 4.4

# thickness of the cavity in meter
h = 1.6 * 10 ** -3

result = pa.design_string(freq, er, h)

print(result)

#normal feed
pa.write_gerber(freq, er, h, 'patch_design_normal_2.4GHz_4.4_er_1.6_h.gbr', 'normal')
#inset feed
pa.write_gerber(freq, er, h, 'patch_design_inset_2.4GHz_4.4_er_1.6_h.gbr', 'inset')
#costumize design result
design = pa.design(freq, er, h)

design.feeder_length *= 1.25

design.feeder_width *= 1.10

pa.write_gerber_design(design, 'custom_patch_normal_design.gbr', 'normal')

pa.write_gerber_design(design, 'custom_patch_inset_design.gbr', 'inset')




import json
import patch_antenna as pa
# resonant frequency in Hz
freq = 2.4 * 10 ** 9
# dielectric constant
er = 4.4
# thickness of the cavity in meter
h = 1.6 * 10 ** -3
result = pa.design(freq, er, h)
# pretty printing
print(json.dumps(result, indent=4))

pa.write_gerber(freq, er, h, 'patch_design_normal_2.4GHz_4.4_er_1.6_h.gbr', 'normal')

pa.write_gerber(freq, er, h, 'patch_design_inset_2.4GHz_4.4_er_1.6_h.gbr', 'inset')


from patch_util.patch import get_directivity, patch_eh_plane_plot, surface_plot, get_i1
freq = 2.4e9
er = 4.4
h = 1.6* 10 ** -3

patch_width = result['patch_width']
i1 = get_i1(patch_width, freq)
print("The value for equation (14-53a) : ", i1)


d1 = get_directivity(patch_width, freq, patch_length)
print('Directivity : ', d1, ' dB')

# Let's assume the value i2
i2 = 2
d2 = get_directivity_two_slot(W, freq, i2)
print("Directivity (two-slot) : ", d2, ' dB')



fields = patch_eh_plane_plot(freq, W, L, h, Er)




#S11 plot
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams.update({'font.size': 14})

with open('simulated_data.txt') as f:
    lines = f.readlines()
    x = [1000*float(line.split()[0]) for line in lines]
    y = [line.split()[1] for line in lines]

#with open('measured_data.txt') as f2:
#    lines_m = f2.readlines()
#    x_m = [float(line2.split()[0])/1000000 for line2 in lines_m]
#    y_m = [line2.split()[1] for line2 in lines_m]

plt.plot(x, y, c='royalblue', zorder=5, linewidth=2, label="$S_{11} \ \mathrm{(Simulated)}$")
#plt.plot(x_m, y_m, c='#2ca02c', zorder=10, linewidth=2, label="$S_{11} \ \mathrm{(Measured)}$")
#plt.xticks(np.arange(200, 3000+1, 200))


plt.xlim(0,3000)
plt.ylim(-25,0)
plt.title('$\mathrm{AcubeSAT \ Patch \ Antenna} \ | \ \mathrm{Reflection \ Coefficient} \ (S_{11})$', y=1.01, fontsize=24)
plt.xlabel('$\mathrm{Frequency \ (MHz)}$', fontsize=20)
plt.ylabel('$\mathrm{Magnitude \ (dB)}$', fontsize=20)

#plt.axvline(x=2400, color='brown', linestyle='--', linewidth=2, zorder=15)
plt.axvline(x=2425, color='brown', linestyle='--', linewidth=2, zorder=15)
#plt.axvline(x=2450, color='brown', linestyle='--', linewidth=2, zorder=15)

plt.fill_between([2400,2450], [-25,-25], facecolor='green', alpha=0.2)

plt.axhline(y=-10, color='black', linestyle='--', zorder=1, linewidth=1)
plt.annotate('$\mathrm{Center \ frequency} \ \ \ \ \ \ \ \ f_{0} = 2.425 \mathrm{\ GHz}$', xy=(750, 7), xycoords='axes points', size=18, ha='left', va='bottom', color='brown')

#plt.annotate('$\mathrm{Band \ of \ interest}$', xy=(750, 20), xycoords='axes points', size=18, ha='left', va='bottom', color='green')

plt.text(23.6, -9.7, '$\mathrm{Fractional \ Bandwidth}$: $\mathrm{10.16 \%}$', color = 'forestgreen', fontsize=18)

plt.text(2313-237, -9.8, '$\mathrm{2313 \ MHz}$', color = 'b', fontsize=18)
plt.scatter(2313, -10, c='royalblue', s=10**2, zorder=10, marker='o')
plt.text(2565+25, -9.8, '$\mathrm{2565 \ MHz}$', color = 'b', fontsize=18)
plt.scatter(2565, -10, c='royalblue', s=10**2, zorder=10, marker='o')


plt.text(24, -24.5, '$\mathrm{Simulated \ with \ a \ Time \ Domain \ Solver.}$', color = 'dimgray', fontsize=18)

#plt.fill_between([-100,3000], [-100,-500], facecolor='green', alpha=0.2)

plt.legend(bbox_to_anchor=(0.00, 0.99), loc="upper left")

plt.grid()

#plt.show()

plt.savefig('s11.png', bbox_inches='tight')