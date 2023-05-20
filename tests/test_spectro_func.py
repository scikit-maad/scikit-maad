import os
import pytest
import numpy as np
from maad.sound import spectrogram

@pytest.fixture
def test_signal():
    # create a test signal
    fs = 44100  # sample rate of 44.1 kHz
    dur = 1.0  # duration of 1 second
    t = np.linspace(0, dur, int(fs*dur), endpoint=False)  # time vector
    f = 1000  # frequency of 1 kHz
    x = np.sin(2*np.pi*f*t)  # sine wave at 1 kHz
    return x, fs

def test_spectrogram_dims(test_signal):
    x, fs = test_signal
    # compute the spectrogram of the test signal
    nperseg = int(fs*0.01)  # window length of 10 ms
    noverlap = int(nperseg/2)  # overlap of 50%
    Sxx,tn,fn,ext = spectrogram(x, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)

    # check that the dimensions of the spectrogram are correct
    assert fn.shape == (nperseg//2,)  # should be (nperseg/2 + 1, )
    assert tn.shape == (Sxx.shape[1],)  # should match the number of columns in Sxx
    assert Sxx.shape == (nperseg//2, len(tn))  # should be (nperseg/2 + 1, len(times))


def test_spectrogram_consistency(test_signal):
    x, fs = test_signal
    EXPECTED_SXX = np.load(os.path.join('tests','data','Sxx.npy'))

    # compute the spectrogram
    Sxx,tn,fn,ext = spectrogram(x, fs=fs, window='hann', nperseg=int(fs*0.01), noverlap=int(fs*0.005))

    # compare the spectrograms
    assert np.allclose(Sxx, EXPECTED_SXX), \
        'Sxx not equal'
    

