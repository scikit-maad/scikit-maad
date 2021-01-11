# -*- coding: utf-8 -*-
""" 
Ensemble of functions to load and preprocess audio signals.
"""

from .io import (load,
                 loadSpectro)

from .filter import (select_bandwidth,
                     fir_filter,
                     sinc,
                     smooth)

from .spectral_subtraction import (remove_background,
                                   remove_background_morpho,
                                   remove_background_along_axis,
                                   median_equalizer)

from .trim import wave2frames

from .envelope import envelope

from .spectrogram import (spectrogram,
                          avg_power_spectro,
                          avg_amplitude_spectro,
                          intoOctave)
                          
from .metrics import (audio_SNR,
                      spectral_SNR,
                      sharpness)

__all__ = [
        # io.py
        'load',
        'loadSpectro',
        # filter.py
        'select_bandwidth',
        'fir_filter',
        'sinc',
        'smooth',
        # spectral_subtraction.py
        'remove_background', 
        'remove_background_morpho',
        'remove_background_along_axis',
        'median_equalizer',
        # trim.py
        'wave2frames',
        # envelope.py
        'envelope',
        # spectrogram.py
        'spectrogram',
        'avg_power_spectro',
        'avg_amplitude_spectro',
        'intoOctave',
        # metrics.py
        'audio_SNR',
        'spectral_SNR',
        'sharpness']