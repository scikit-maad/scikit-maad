import os
import numpy as np
import pandas as pd
import pytest
from maad.rois import template_matching
from maad import sound, util

@pytest.fixture
def sample_data():
    # Set spectrogram parameters
    tlims = (9.8, 10.5)
    flims = (6000, 12000)
    nperseg = 1024
    noverlap = 512
    window = 'hann'
    db_range = 80
    peak_th = 0.3

    # load data
    s, fs = sound.load(os.path.join('data','spinetail.wav'))

    # Compute spectrogram for template signal
    Sxx_template, _, _, _ = sound.spectrogram(s, fs, window, nperseg, noverlap, flims, tlims)
    Sxx_template = util.power2dB(Sxx_template, db_range)

    # Compute spectrogram for target audio
    Sxx_audio, tn, _, ext = sound.spectrogram(s, fs, window, nperseg, noverlap, flims)
    Sxx_audio = util.power2dB(Sxx_audio, db_range)

    return Sxx_audio, Sxx_template, tn, ext, peak_th

def test_template_matching_output(sample_data):
    Sxx_audio, Sxx_template, tn, ext, peak_th = sample_data
    EXPECTED = np.load(os.path.join('tests','data','template_matching_data.npz'), allow_pickle = True)
    xcorrcoef, rois = template_matching(Sxx_audio, Sxx_template, tn, ext, peak_th)
    
    # take snapshot -- np.savez('data/template_matching_data.npz', xcorrcoef=xcorrcoef, rois=rois.to_numpy())

    # Compare the actual result with the saved snapshot
    assert np.allclose(xcorrcoef, EXPECTED['xcorrcoef']), \
        'Cross-correlation not equal'
    assert np.allclose(rois.to_numpy(), EXPECTED['rois']), \
        'ROIs detected not equal'