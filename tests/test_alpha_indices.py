#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for alpha acoustic indices

"""

import os
import pytest
import maad
import numpy as np


#%% Temporal indices

def test_temporal_median():
    # Compute temporal median
    s, fs = maad.sound.load('../data/spinetail.wav')
    temporal_median_index = maad.features.temporal_median(s)

    # Load expected values
    expected_values = 0.007934564717486147
    
    assert np.allclose(temporal_median_index, expected_values)
    
def test_temporal_entropy():
    # Compute temporal median
    s, fs = maad.sound.load('../data/spinetail.wav')
    temporal_entropy_index = maad.features.temporal_entropy(s)

    # Load expected values
    expected_values = 0.7518917279549968
    
    assert np.allclose(temporal_entropy_index, expected_values)

def test_temporal_activity():
    # Compute temporal median
    s, fs = maad.sound.load('../data/spinetail.wav')
    temporal_activity_index = maad.features.temporal_activity(s, dB_threshold=6)

    # Load expected values
    expected_values = (0.36838978015448604, 620, 24.41347313621499)
    
    assert np.allclose(temporal_activity_index, expected_values)


#%% Spectral indices

def test_spectral_entropy():
    # Compute spectral entropy
    s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    Sxx_power, tn, fn, _ = maad.sound.spectrogram (s, fs)  
    spectral_entropy_indices = maad.features.spectral_entropy(
        Sxx_power, fn, flim=(2000,10000))

    # Load expected values
    expected_values = (0.26807540760637083, 
                       0.4883515296525619,
                       0.24146196318401636,
                       1,
                       17.580495975968567,
                       3.5452699010615505)
    
    assert np.allclose(spectral_entropy_indices,expected_values)
    

#%% Spectro-temporal indices