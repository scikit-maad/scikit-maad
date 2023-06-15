#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for shape features

"""
import os
import numpy as np
import pandas as pd
from maad import sound, features, util


def test_shape_features_low():
    EXPECTED_SHAPE = np.load(os.path.join('tests','data','shape_spinetail_res_low.npy'))
    EXPECTED_PARAMS = np.load(os.path.join('tests','data','params_res_low.npy'))
    s, fs = sound.load(os.path.join('data','spinetail.wav'))
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, db_range=100)
    Sxx_db = util.power2dB(Sxx, db_range=100)
    shape, params = features.shape_features(Sxx_db, resolution='low')
    assert np.allclose(shape.values, EXPECTED_SHAPE), \
        'shape not equal'
    assert np.allclose(params.values, EXPECTED_PARAMS), \
        'params not equal'

def test_shape_features_med():
    EXPECTED_SHAPE = np.load(os.path.join('tests','data','shape_spinetail_res_med.npy'))
    EXPECTED_PARAMS = np.load(os.path.join('tests','data','params_res_med.npy'))
    s, fs = sound.load(os.path.join('./data/','spinetail.wav'))
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, db_range=100)
    Sxx_db = util.power2dB(Sxx, db_range=100)
    shape, params = features.shape_features(Sxx_db, resolution='med')
    assert np.allclose(shape.values, EXPECTED_SHAPE), \
        'shape not equal'
    assert np.allclose(params.values, EXPECTED_PARAMS), \
        'params not equal'
    
def test_shape_features_high():
    EXPECTED_SHAPE = np.load(os.path.join('tests','data','shape_spinetail_res_high.npy'))
    EXPECTED_PARAMS = np.load(os.path.join('tests','data','params_res_high.npy'))
    s, fs = sound.load(os.path.join('./data/','spinetail.wav'))
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, db_range=100)
    Sxx_db = util.power2dB(Sxx, db_range=100)
    shape, params = features.shape_features(Sxx_db, resolution='high')
    assert np.allclose(shape.values, EXPECTED_SHAPE), \
        'shape not equal'
    assert np.allclose(params.values, EXPECTED_PARAMS), \
        'params not equal'