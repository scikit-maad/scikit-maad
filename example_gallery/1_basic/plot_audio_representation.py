#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio representation
====================

An audio signal can be represented in both, temporal and spectral domains. 
These representations are complementary and fundamental to understand the audio
signal characteristics. In this introductory example we will load an audio signal, 
apply basic transformations to better understand its features in time and frequency.
"""

#%% 
import matplotlib.pyplot as plt
from maad import sound
from maad import util

# Load an audio file and plot the waveform
signal, sample_rate = sound.load(
    filename='../../data/spinetail.wav'
)

# Plot and show the waveform
util.plot_wave(
    s=signal, 
    fs=sample_rate
)
plt.show()

#%% 
# It can be noticed that in this audio there are four consecutive songs of the spinetail 
# *Cranioleuca erythorps*, every song lasting of approximatelly two seconds. 

# Trim the signal to zoom in on the details of the song.
signal_trimmed = sound.trim(
    s=signal, 
    fs=sample_rate, 
    min_t=5, 
    max_t=8
)

#%% 
# Compute the envelope of the trimmed signal
envelope = sound.envelope(
    s=signal_trimmed, 
    mode='fast', 
    Nt=128
)

# Compute the power spectrum density estimate using the Fourier transform (fft)
psd_estimate, sample_frequency_indices = sound.spectrum(
    s=signal,
    fs=sample_rate,
    nperseg=1024,
    method='welch',
)

# Compute the power spectrogram using the short-time Fourier transform (stft)
Sxx_power, tn, fn, box_extend = sound.spectrogram(
    x=signal_trimmed,
    fs=sample_rate,
    window='hann',
    nperseg=1024,
    noverlap=512,
)

#%% 
# Visualize the signal characteristics in the temporal and spectral domains
figure, axes = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(8, 10),
)
util.plot_wave(
    s=signal_trimmed,
    fs=sample_rate,
    ax=axes[0],
)
util.plot_wave(
    s=envelope,
    fs=sample_rate,
    ax=axes[1],
)
util.plot_spectrum(
    pxx=psd_estimate,
    f_idx=sample_frequency_indices,
    ax=axes[2],
)
util.plot_spectrogram(
    Sxx=Sxx_power,
    extent=box_extend,
    ax=axes[3],
    colorbar=False,
)
plt.show()