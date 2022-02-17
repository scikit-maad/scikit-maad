#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for preprocessing functions

"""
import os
import pytest
import maad
import pandas as pd
import numpy as np
from maad import sound
from maad.rois import find_rois_cwt


def test_find_rois_cwt():
    DATA = pd.read_csv(os.path.join('data','find_rois_cwt_trill.csv'))
    s, fs = sound.load(os.path.join('..','data','spinetail.wav'))
    df = find_rois_cwt(s, fs, flims=(4500,8000), tlen=2, th=0, display=False)
    assert np.allclose(df,DATA)