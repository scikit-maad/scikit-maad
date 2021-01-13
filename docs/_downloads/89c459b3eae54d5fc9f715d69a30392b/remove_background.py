#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remove background noise from audio with signal processing tools
===============================================================

This example shows different ways to remove background noise directly from
the spectrogram.
We use the sharpness metric to have a quantitative estimation of how well is 
the noise reduction. This metric gives partial information. Other metrics 
should be use in complement.

"""
# sphinx_gallery_thumbnail_path = '../_images/sphx_glr_remove_background.png'

from maad.sound import load, spectrogram
from maad.util import plot2D, power2dB
from maad.rois import (remove_background, median_equalizer, 
                       remove_background_morpho, 
                       remove_background_along_axis, sharpness)
import numpy as np

from timeit import default_timer as timer

import matplotlib.pyplot as plt

#%%
# First, we load the audio file and take its spectrogram.
# The linear spectrogram is then transformed into dB. The dB range is  96dB 
# which is the maximum dB range value for a 16bits audio recording. We add
# 96dB in order to get have only positive values in the spectrogram
s, fs = load('../data/tropical_forest_morning.wav')
#s, fs = load('../data/cold_forest_night.wav')
Sxx, tn, fn, ext = spectrogram(s, fs, fcrop=[0,20000], tcrop=[0,60])
Sxx_dB = power2dB(Sxx, db_range=96) + 96

#%%
# We plot the original spectrogram.
fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, sharex=True)
plot2D(Sxx_dB, ax=ax0, extent=ext, title='original', xlabel=None,
       vmin=np.median(Sxx_dB), vmax=np.median(Sxx_dB)+40)

print ("Original sharpness : %2.3f" % sharpness(Sxx_dB))

#%%
# Test the function "remove_background"
start = timer()
X1, noise_profile1, _ = remove_background(Sxx_dB)
elapsed_time = timer() - start
print("---- test remove_background -----")
print("duration %2.3f s" % elapsed_time)
print ("sharpness : %2.3f" % sharpness(X1))

plot2D(X1, ax=ax1, extent=ext, title='remove_background', xlabel=None,
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

plot2D(X2, ax=ax2, extent=ext, title='median_equalizer', xlabel=None,
       vmin=np.median(X2), vmax=np.median(X2)+40)

plot2D(X2,extent=ext, title='median_equalizer',
       vmin=np.median(X2), vmax=np.median(X2)+40)

#%%
# Test the function "remove_background_morpho"
start = timer()
X3, noise_profile3,_ = remove_background_morpho(Sxx_dB, q=0.95) 
elapsed_time = timer() - start
print("---- test remove_background_morpho -----")
print("duration %2.3f s" % elapsed_time)
print ("sharpness : %2.3f" %sharpness(X3))

plot2D(X3, ax=ax3, extent=ext, title='remove_background_morpho', xlabel=None, 
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

plot2D(X4, extent=ext, title='remove_background_along_axis',
       vmin=np.median(X4), vmax=np.median(X4)+40)

plot2D(X4, ax=ax4, extent=ext, title='remove_background_along_axis',
       vmin=np.median(X4), vmax=np.median(X4)+40)

plt.tight_layout()
