#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for shape features

"""
import os
import numpy as np
import pandas as pd
from maad import sound, features, util


def test_shape_features():
    DATA = pd.read_csv(os.path.join('./tests/data/','shape_spinetail_res_low.csv'))
    s, fs = sound.load(os.path.join('./data/','spinetail.wav'))
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, db_range=100)
    Sxx_db = util.power2dB(Sxx, db_range=100)
    shape, params = features.shape_features(Sxx_db, resolution='low')
    assert np.allclose(shape,DATA)
