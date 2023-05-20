import numpy as np
from maad.sound.transform import resample

def test_resample():
    # Generate test signal of 1 second at 44100 Hz
    sr_orig = 44100
    duration = 1.0
    freq = 440.0
    t = np.linspace(0, duration, int(sr_orig * duration), endpoint=False)
    x = np.sin(2 * np.pi * freq * t)
    
    # Test 1: Check if resampling reduces the size of the signal by half
    sr_new = 22050
    y = resample(x, sr_orig, sr_new, res_type='scipy')
    assert len(y) == len(x) // 2

    # Test 2: Resampling to higher sampling rate (88200 Hz) and back to original
    sr_new = 88200
    y = resample(x, sr_orig, sr_new, res_type='scipy')
    assert len(y) == int(len(x) * sr_new / sr_orig)
    z = resample(y, sr_new, sr_orig, res_type='scipy')
    assert np.allclose(x, z, rtol=1e-02, atol=1e-02, equal_nan=False)


    # Test 3: Check if resampling a signal with same sample rate returns idential signal
    w = resample(x, sr_orig, sr_orig)
    assert np.allclose(x, w, rtol=1e-04, atol=1e-08, equal_nan=False)
