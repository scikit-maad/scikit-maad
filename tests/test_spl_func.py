#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test SPL module

"""

import numpy as np
import os
import pandas as pd
from maad import sound, spl, util
import pytest

#%% psd2leq

def test_psd2leq():
    w, fs = sound.load('./data/cold_forest_daylight.wav') 
    Sxx_power,_,_,_ = sound.spectrogram (w, fs)
    S_power_mean = sound.avg_power_spectro(Sxx_power) 
    values = spl.psd2leq(S_power_mean, gain=42)

    # Load expected values
    expected_values = 47.537824826354665

    assert np.allclose(values, expected_values)

#%% wav2leq

def test_wav2leq():
    w, fs = sound.load('./data/cold_forest_daylight.wav') 
    Leq = spl.wav2leq (w, fs, gain=42)  
    values =  util.mean_dB(Leq)
        
    # Load expected values
    expected_values = 48.55488267086038

    assert np.allclose(values, expected_values)