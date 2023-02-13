#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for preprocessing functions

"""
import pandas as pd
import numpy as np
from maad import sound
from maad.rois import find_rois_cwt
from tests._paths import DATA_PATH
from tests._paths import TESTS_DATA_PATH


def test_find_rois_cwt():
    # Arrange
    data = pd.read_csv(str(TESTS_DATA_PATH / 'find_rois_cwt_trill.csv'))
    s, fs = sound.load(str(DATA_PATH / 'spinetail.wav'))

    # Act
    df = find_rois_cwt(s, fs, flims=(4500, 8000), tlen=2, th=0, display=False)

    # Assert
    assert np.allclose(df, data)
