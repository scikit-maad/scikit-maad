"""
Test module for acoustic traits

"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from maad import sound, features

#%% Sepectral traits

def test_spectra_features():
    s, fs = sound.load('./data/spinetail.wav')

    spectral_features = features.all_spectral_features(s, fs)

    expected_values = pd.DataFrame({"sm":2.2763302268911505e-06,
                                    "sv":8.11804174130019e-11,
                                    "ss":5.8446635093417045,
                                    "sk":40.488906373539436,
                                    "Freq 5%":6029.296875,
                                    "Freq 25%":6416.89453125,
                                    "Freq 50%":6632.2265625,
                                    "Freq 75%":6890.625,
                                    "Freq 95%":9216.2109375,
                                    "peak_freq":6632.2265625,
                                    "Time 5%":1.2190476190476192,
                                    "Time 25%":5.712108843537415,
                                    "Time 50%":11.818956916099774,
                                    "Time 75%":16.555827664399093,
                                    "Time 95%":17.751655328798186,
                                    "bandwidth_50":473.73046875,
                                    "bandwidth_90":3186.9140625,
                                    "bandwidth_3dB":320.8791300822477}, index=[0])

    assert_frame_equal(spectral_features, expected_values)

#%% Temporal traits

def test_temporal_features():
    s, fs = sound.load('./data/spinetail.wav')

    temporal_features = features.all_temporal_features(s, fs)

    expected_values = pd.DataFrame({"sm":-2.04326427e-19,
                                    "sv":0.0011670735714956646,
                                    "ss":-0.006547980427883208,
                                    "sk":24.711610834321217,
                                    "Time 5%":1.1361585719559142,
                                    "Time 25%":3.067628144280968,
                                    "Time 50%":11.702433291145915,
                                    "Time 75%":15.678988292991615,
                                    "Time 95%":17.724073722512262,
                                    "zcr":10500.397192384766,
                                    "duration_50":12.611360148710647,
                                    "duration_90":16.58791515055635}, index=[0])

    assert_frame_equal(temporal_features, expected_values)

#%%
def test_pulse_rate():
  s, fs = sound.load('./data/spinetail.wav')

  events, amp = features.pulse_rate(s, fs, dmin=0.5, threshold1=15)

  expected_values = pd.DataFrame({"min_t":[0.7546123372558226, 5.3882692235847225, 11.467152433856802, 16.318792339259353],
                                  "max_t":[2.6254293310975085, 7.209350613619326, 13.260147831512008, 17.956873120130904],
                                  "duration":[1.8708169938416859, 1.8210813900346032, 1.792995397655206, 1.638080780871551]},
                                  index=[0,1,2,3])

  assert_frame_equal(events, expected_values)