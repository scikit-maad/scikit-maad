#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for alpha acoustic indices

"""

import numpy as np
import os
import pandas as pd
from maad import sound, features
import pytest

#%% Temporal indices

def test_temporal_median():
    # Compute temporal median
    s, fs = sound.load('./data/spinetail.wav')
    temporal_median_index = features.temporal_median(s)

    # Load expected values
    expected_values = 0.007934564717486147

    assert np.allclose(temporal_median_index, expected_values)

def test_temporal_entropy():
    # Compute temporal median
    s, fs = sound.load('./data/spinetail.wav')
    temporal_entropy_index = features.temporal_entropy(s)

    # Load expected values
    expected_values = 0.7518917279549968

    assert np.allclose(temporal_entropy_index, expected_values) # type: ignore

def test_temporal_activity():
    # Compute temporal median
    s, fs = sound.load('./data/spinetail.wav')
    temporal_activity_index = features.temporal_activity(s, dB_threshold=6)

    # Load expected values
    expected_values = (0.36838978015448604, 620, 24.41347313621499)

    assert np.allclose(temporal_activity_index, expected_values) # type: ignore


#%% Spectral indices

def test_spectral_entropy():
    # Compute spectral entropy
    s, fs = sound.load('./data/cold_forest_daylight.wav')
    Sxx_power, tn, fn, _ = sound.spectrogram (s, fs)
    spectral_entropy_indices = features.spectral_entropy(
        Sxx_power, fn, flim=(2000,10000))

    expected_values = (
        0.26807540760637083,
        0.488351529652562,
        0.24146196318401625,
        0.2536420606912331,
        17.580495975968567,
        3.5452699010615505)

    assert np.allclose(spectral_entropy_indices,expected_values) # type: ignore

#%% Spectro-temporal indices

EXPECTED_VALUES = np.load(os.path.join('tests','data','indices_temporal_values.npy'), allow_pickle=True)
EXPECTED_NAMES = np.load(os.path.join('tests','data','indices_temporal_names.npy'), allow_pickle=True)
EXPECTED_DATAFRAME = pd.DataFrame(data=EXPECTED_VALUES, columns=EXPECTED_NAMES)

@pytest.mark.parametrize(
    "test_filename, expected",
    [
        ('cold_forest_daylight.wav',    EXPECTED_DATAFRAME.iloc[[0]]),
        ('cold_forest_night.wav',       EXPECTED_DATAFRAME.iloc[[1]]),
        ('rock_savanna.wav',            EXPECTED_DATAFRAME.iloc[[2]]),
        ('tropical_forest_morning.wav', EXPECTED_DATAFRAME.iloc[[3]]),
    ]
)

def test_temporal_alpha_indices(test_filename, expected):

    SENSITIVITY = -35         
    GAIN = 26+16       
    THRESHOLD = 3
    REJECT_DURATION = 0.01

    wave,fs = sound.load(filename = os.path.join('data',test_filename))

    wave = sound.trim(wave, fs, 0, 10)

    output = features.all_temporal_alpha_indices(wave, fs, 
                                            gain            = GAIN, 
                                            sensibility     = SENSITIVITY,
                                            dB_threshold    = THRESHOLD, 
                                            rejectDuration  = REJECT_DURATION)
    
    output.insert(0, "filename", test_filename)
    
    for param in list(output) :
        # test if it's a string
        if isinstance(expected[param].to_numpy()[0], str) :
            assert output[param].to_numpy()[0] == expected[param].to_numpy()[0], \
                '{} : output value {} is not equal to expected value {}'.format(param, output[param].values,expected[param].values)
        else :
            assert np.isclose(output[param].to_numpy()[0], expected[param].to_numpy()[0]), \
                '{} : output value {} is not equal to expected value {}'.format(param, output[param].values,expected[param].values)


# %%
EXPECTED_VALUES = np.load(os.path.join('tests','data','indices_spectral_values.npy'), allow_pickle=True)
EXPECTED_NAMES = np.load(os.path.join('tests','data','indices_spectral_names.npy'), allow_pickle=True)
EXPECTED_DATAFRAME = pd.DataFrame(data=EXPECTED_VALUES, columns=EXPECTED_NAMES)

@pytest.mark.parametrize(
    "test_filename, expected",
    [
        ('cold_forest_daylight.wav',    EXPECTED_DATAFRAME.iloc[[0]]),
        ('cold_forest_night.wav',       EXPECTED_DATAFRAME.iloc[[1]]),
        ('rock_savanna.wav',            EXPECTED_DATAFRAME.iloc[[2]]),
        ('tropical_forest_morning.wav', EXPECTED_DATAFRAME.iloc[[3]]),
    ]
)

def test_spectral_alpha_indices(test_filename, expected):

    SENSITIVITY = -35         # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)   
    GAIN = 26+16       # Amplification gain (26dB (SM4 preamplifier))
    WINDOW = 'hann'
    N_FFT = 1024
    FLIM_LOW = [0,1000]
    FLIM_MED = [1000,6000]
    FLIM_HIG = [6000,20000]
    MASK_PARAM1 = 6
    MASK_PARAM2 = 0.5

    wave,fs = sound.load(filename = os.path.join('data',test_filename))

    wave = sound.trim(wave, fs, 0, 10)

    Sxx_power,tn,fn,ext = sound.spectrogram (wave, fs, 
                                            window      =WINDOW, 
                                            nperseg     = N_FFT, 
                                            noverlap    =N_FFT//2)   
    
    output, _ = features.all_spectral_alpha_indices(Sxx_power,
                                                    tn,fn,
                                                    flim_low = FLIM_LOW, 
                                                    flim_mid = FLIM_MED, 
                                                    flim_hi  = FLIM_HIG, 
                                                    gain = GAIN, 
                                                    sensitivity = SENSITIVITY,
                                                    R_compatible = 'soundecology',
                                                    mask_param1 = MASK_PARAM1, 
                                                    mask_param2=MASK_PARAM2)
    
    output.insert(0, "filename", test_filename)
    
    for param in list(output) :
        # test if it's a string
        if isinstance(expected[param].to_numpy()[0], str) :
            assert output[param].to_numpy()[0] == expected[param].to_numpy()[0], \
                '{} : output value {} is not equal to expected value {}'.format(param, output[param].values,expected[param].values)
        else :
            assert np.isclose(output[param].to_numpy()[0], expected[param].to_numpy()[0]), \
                '{} : output value {} is not equal to expected value {}'.format(param, output[param].values,expected[param].values)


