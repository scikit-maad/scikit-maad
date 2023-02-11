#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for shape features

"""
import numpy as np
import pandas as pd
from maad import sound, features, util
from tests._paths import DATA_PATH
from tests._paths import TESTS_DATA_PATH


def test_shape_features():
    data = pd.read_csv(str(TESTS_DATA_PATH / 'shape_spinetail_res_low.csv'))
    s, fs = sound.load(str(DATA_PATH / 'spinetail.wav'))
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, db_range=100)
    Sxx_db = util.power2dB(Sxx, db_range=100)
    shape, params = features.shape_features(Sxx_db, resolution='low')
    assert np.allclose(shape, data)
