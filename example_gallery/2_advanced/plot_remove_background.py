#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remove background noise with signal processing tools
====================================================

Environmental audio recordings usually have stationary noise that needs to be removed to
enhance the signal to noise ratio of biological sounds.
This example shows different ways to remove stationary background noise using spectral 
subtraction techniques. These techniques are applied over the spectrogram and return a 2D matrix. 
We use the sharpness metric to have a quantitative estimation of how well is the noise 
reduction. For a more comprehensive analysis, other metrics should be use in complement.

"""
# sphinx_gallery_thumbnail_path = './_images/sphx_glr_plot_remove_background_002.png'

from maad.util import plot2d, power2dB
from maad.sound import (load, spectrogram, 
                       remove_background, median_equalizer, 
                       remove_background_morpho, 
                       remove_background_along_axis, sharpness)
import numpy as np

from timeit import default_timer as timer

import matplotlib.pyplot as plt

#%%
# Load and plot the spectrogram of the original audio file
# --------------------------------------------------------
# First, we load the audio file and take its spectrogram.
# The linear spectrogram is then transformed into dB. The dB range is  96dB 
# which is the maximum dB range value for a 16bits audio recording. We add
# 96dB in order to get have only positive values in the spectrogram.
s, fs = load('../../data/tropical_forest_morning.wav')
Sxx, tn, fn, ext = spectrogram(s, fs, fcrop=[0,20000], tcrop=[0,60])
Sxx_dB = power2dB(Sxx, db_range=96) + 96
plot2d(Sxx_dB, extent=ext, title='original',
       vmin=np.median(Sxx_dB), vmax=np.median(Sxx_dB)+40)

print ("Original sharpness : %2.3f" % sharpness(Sxx_dB))

#%%
# Test different methods to remove stationary background noise
# ------------------------------------------------------------
# Test the function "remove_background"
start = timer()
X1, noise_profile1, _ = remove_background(Sxx_dB)
elapsed_time = timer() - start
print("---- test remove_background -----")
print("duration %2.3f s" % elapsed_time)
print ("sharpness : %2.3f" % sharpness(X1))

plot2d(X1, extent=ext, title='remove_background',
       vmin=np.median(X1), vmax=np.median(X1)+40)

#%%
# Test the function "median_equalizer"
start = timer()
X2 = median_equalizer(Sxx)
X2 = power2dB(X2)
elapsed_time = timer() - start
print("---- test median_equalizer -----")
print("duration %2.3f s" % elapsed_time)
print ("sharpness : %2.3f" %sharpness(X2))

plot2d(X2,extent=ext, title='median_equalizer',
       vmin=np.median(X2), vmax=np.median(X2)+40)

#%%
# Test the function "remove_background_morpho"
start = timer()
X3, noise_profile3,_ = remove_background_morpho(Sxx_dB, q=0.95) 
elapsed_time = timer() - start
print("---- test remove_background_morpho -----")
print("duration %2.3f s" % elapsed_time)
print ("sharpness : %2.3f" %sharpness(X3))

plot2d(X3, extent=ext, title='remove_background_morpho',
       vmin=np.median(X3), vmax=np.median(X3)+40)

#%%
# Test the function "remove_background_along_axis"
start = timer()
X4, noise_profile4 = remove_background_along_axis(Sxx_dB,mode='median', axis=1) 
#X4 = power2dB(X4) 
elapsed_time = timer() - start
print("---- test remove_background_along_axis -----")
print("duration %2.3f s" % elapsed_time)
print ("sharpness : %2.3f" %sharpness(X4))

plot2d(X4,  extent=ext, title='remove_background_along_axis',
       vmin=np.median(X4), vmax=np.median(X4)+40)

plt.tight_layout()
