#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detection distance estimation 
==============================

When sound travels in air (or in water), initial acoustic signature may change 
regarding distances due to attenuation. Sound attenuation in natural environment 
is very complex to model. We propose to reduce the complexity by decomposing 
the sound attenuation to three main sources of attenuation : 
- the geometric attenuation (Ageo) also known as spreading loss or geometric 
dispersion, 
- the atmospheric attenuation (Aatm) 
- and the habitat attenuation (Ahab). The later encompasses several sources of 
attenuation and might be seen as a proxy.

Knowing the attenuation law and the sound level at each frequency, it is 
possible to estimate the detection distance of each frequency component of 
that sound

"""

from maad import sound, spl, util
import numpy as np

#%%
# In this example, we will estimate the detection distance of each frequency
# component of a sound that travels in rainforest at noon, when the ambient
# sound level is the lowest. 

# Let's decide that the propagation sound is a wideband sound ranging from 250Hz
# to 20000Hz with an initial sound pressure level of 80 dBSPL @ 1m which 
# corresponds to the sum of sound pressure level along the full frequency 
# bandwidth.

# frequency vector from 250Hz to 20000Hz
f = np.arange(0,20000,500)
# From 80dB SPL to the repartition of the sound pressure level at each 
# frequency component
L0 = 80
L0_per_bin = spl.dBSPL_per_bin(L0,f)
# Distance in m at which the initial sound pressure level is measured
r0 = 1

# Sound pressure level of the background noise (or ambient sound) at each 
# frequency component (from 250Hz to 20kHz). This was measured experimentaly
# in a rainforest (French Guiana) at noon.
L_bkg = np.array([44.270917, 30.191839, 27.586848, 27.333053, 25.608430, 25.805781, 
                  23.205826, 21.682666, 20.631086, 22.437462, 24.080126, 21.654460, 
                  19.032034, 26.467840, 33.455814, 45.779824, 44.420644, 29.945157, 
                  19.751421, 15.266170, 11.932672, 10.403127,  9.641225,  8.981515,  
                  8.075566,  7.311595,  7.447614,  7.263082,  6.991958,  6.811319, 
                  7.854252, 13.038273, 11.911974,  9.953524,  4.192154,  3.494721,  
                  3.234791,  3.056770,  2.936258,  2.696155 ])


# decimate the number of value by 2 for display
f = f[np.arange(0,len(f),2)]
L0_per_bin = L0_per_bin[np.arange(0,len(L0_per_bin),2)]
L_bkg = L_bkg[np.arange(0,len(L_bkg),2)]

# Estimate the detection distance for each frequency
f, r = spl.detection_distance (L_bkg, L0_per_bin, f, r0, t = 24, rh = 87)

#%%
# Display the detection distance for each frequency as a bar plot

# import the graphic library
import matplotlib.pyplot as plt

# Define a function to add value labels
def valuelabel(f,r):
    for i in range(len(f)):
        plt.text(i,
                 r[i]+2,
                 '%dm' % r[i], 
                 ha = 'center',
                 size=6,
                 bbox = dict(facecolor = 'grey', 
                             alpha =0.5),)
  
# Create the figure and set the size
fig = plt.figure()
fig.set_figwidth(8)
fig.set_figheight(7)

# Create bars
y_pos = np.arange(len(f))
plt.bar(y_pos, r)

# Create names on the x-axis
plt.xticks(y_pos, f)
plt.xticks(rotation=45)

# Call function
valuelabel(f, r) 

# Define labels
plt.ylabel("Detection distance [m]")
plt.xlabel("Frequency [Hz]")

# Show graphic
plt.grid(axis='y')
plt.show()