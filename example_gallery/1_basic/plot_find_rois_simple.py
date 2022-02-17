#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple audio segmentation
=========================

In audio signals, regions of interest are usually regions with high density of energy. The function ``find_rois_cwt`` allows finding regions of interest in the signal giving very simple and intuitive parameters: temporal length and frequency limits. This segmentation can be seen as a coarse detection process, the starting point of more advanced classification methods.

The following sound example as two main different soundtypes in the foreground:

- A bouncy trill between 4.5 and 8 kHz lasting approximately 2 seconds.
- A fast descending chirp between 8 and 12 kHz lasting 0.1 approximately seconds.
"""

#%% 
# Load audio file
# ---------------
# Load an audio file and compute the spectrogram for visualization.

from maad import sound
from maad.rois import find_rois_cwt
from maad.util import plot_spectrogram

s, fs = sound.load('../../data/spinetail.wav')
Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=512)
plot_spectrogram(Sxx, extent=ext, db_range=60, gain=20, figsize=(4,10))

#%% 
# Detect the bouncy trill
# -----------------------
# The accelerating trill is the song of a small neotropical bird, the Red-faced Spinetail *Cranioleuca erythrops*. This song can be detected on the recording using the function ``find_rois_cwt`` and setting frequency limits ``flims=(4500,8000)`` and temporal length of signal ``tlen=2``. The segmentation results are returned as a dataframe with temporal segmentation given by the function and using the frequency limits defined by the user.

df_trill = find_rois_cwt(s, fs, flims=(4500,8000), tlen=2, th=0, display=True, figsize=(10,6))
print(df_trill)

#%%
# Detect the fast descending chirp
# --------------------------------
# Alternatively, the fast descending chirp (unknown species) can be segmented in the recording by changing the detection parameters, ``flims`` and ``tlen``.

df_chirp = find_rois_cwt(s, fs, flims=(8000,12000), tlen=0.1, th=0.001, display=True, figsize=(10,6))
print(df_chirp)
