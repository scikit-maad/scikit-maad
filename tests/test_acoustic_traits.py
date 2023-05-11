"""
Test module for acoustic traits

"""
import os
import numpy as np
from maad import sound, features

#%% Spectral features
def test_spectral_features():
    s, fs = sound.load('./data/spinetail.wav')
    spectral_features = features.all_spectral_features(s, fs, nperseg=1024, roi=None)
    EXPECTED_SPECTRAL = np.load(os.path.join('tests','data','spectral_features.npy'))
    assert np.allclose(spectral_features.values, EXPECTED_SPECTRAL)

#%% Temporal features
def test_temporal_features():
    s, fs = sound.load('./data/spinetail.wav')
    temporal_features = features.all_temporal_features(s, fs)
    EXPECTED_TEMPORAL = np.load(os.path.join('tests','data','temporal_features.npy'))
    assert np.allclose(temporal_features.values, EXPECTED_TEMPORAL)
