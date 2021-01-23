#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment audio using predetermined temporal length and frequency limits
======================================================================

In an audio signal, regions of interest are usually regions with high density of energy. The function find_rois_cwt allows finding regions of interest in the signal giving very simple and intuitive parameters: temporal length and frequency limits. This segmentation can be seen as a coarse detection process, the starting point of more advanced classification methods.

The following sound example as two main different soundtypes in the foreground:

- An accelerating trill between 4.5 and 8 kHz lasting approximately 2 seconds
- A fast descending chirp between 8 and 12 kHz lasting 0.1 approximately seconds
"""

#%% Load an audio file and compute the spectrogram for visualization.

from maad import sound
from maad.rois import find_rois_cwt
from maad.util import power2dB, plot2D


s, fs = sound.load('../../data/spinetail.wav')
Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=512)
Sxx_db = power2dB(Sxx, db_range=100) + 100
plot2D(Sxx_db, **{'extent':ext})

#%% 
# Detect the accelerating trill
# -----------------------------
# The accelerating trill is the song of a small neotropical bird, Cranioleuca erythrops. This song can be detected on the recording using the function find_rois_cwt and setting frequency limits flims=(4500,8000) and temporal length of signal tlen=2.

_ = find_rois_cwt(s, fs, flims=(4500,8000), tlen=2, th=0, display=True, figsize=(13,6))

#%%
# Detect the fast descending chirp
# --------------------------------
# Alternatively, the fast descending chirp (unknown species) can be segmented in the recording by changing the detection parameters.

df = find_rois_cwt(s, fs, flims=(8000,12000), tlen=0.1, th=0.001, display=True, figsize=(13,6))

#%%
# The segmentation results are returned as a dataframe with temporal segmentation given by the function and using the frequency limits defined by the user.

print(df)
