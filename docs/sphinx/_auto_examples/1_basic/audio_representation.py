#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:40:28 2021

@author: jsulloa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio representation
====================

An audio signal can be represented in both, temporal and spectral domains. 
These representations give valuable information related to the signal characteristics
and hence are complementary. In this introductory example we will load an audio signal, 
apply basic transformations to better understand its features.
"""

#%% Load an audio file and plot the waveform
import matplotlib.pyplot as plt
from maad import sound
from maad import util

s, fs = sound.load('../../data/spinetail.wav')
util.plot_wave(s, fs)

#%% 
# It can be noticed that in this audio there are four consecutive songs of the spinetail 
# *Cranioleuca erythorps*, every song lasting of approximatelly two seconds. 
# Let's trim the signal to zoom in a single song.

s_trim = sound.trim(s, fs, 5, 8)

#%% Onced trimmed, lets compute the envelope of the signal, and the Fourier and 
# short-time Fourier transforms.
env = sound.envelope(s_trim, mode='fast', Nt=128)
pxx, fidx = sound.spectrum(s, fs, nperseg=1024, method='welch')
Sxx, tn, fn, ext = sound.spectrogram(s_trim, fs, window='hann', nperseg=1024, noverlap=512)

#%% 
# Finally, we can visualize the signal characteristics in the temporal and 
# spectral domains.

fig, ax = plt.subplots(4,1, figsize=(10,12))
util.plot_wave(s_trim, fs, ax=ax[0])
util.plot_wave(env, fs/128, ax=ax[1])
util.plot_spectrum(pxx, fidx, ax=ax[2])
util.plot_spectrogram(Sxx, extent=ext, ax=ax[3], colorbar=False)
