import numpy as np
import pytest
import os
from maad import sound, rois, util

@pytest.fixture
def example_spectrogram():
    # Load example audio file and compute spectrogram
    s, fs = sound.load(os.path.join('data','spinetail.wav'))
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=512, noverlap=256)
    Sxx_db = util.power2dB(Sxx, db_range=80)
    return Sxx_db, tn, fn, ext

def test_spectrogram_local_max(example_spectrogram):
    # Unpack the example spectrogram
    Sxx_db, tn, fn, ext = example_spectrogram
    EXPECTED = np.load(os.path.join('tests','data','local_max.npy'), allow_pickle = True)
    # Test with example parameters
    peak_time, peak_freq = rois.spectrogram_local_max(
        Sxx_db, tn, fn, ext, min_distance=1, threshold_abs=-40, display=False)

    # Create a dictionary with the actual results
    actual_result = np.array([peak_time, peak_freq])

    # Compare the actual result with the saved snapshot
    assert np.allclose(actual_result, EXPECTED), \
        'spectrogram peaks not equal'

    