# -*- coding: utf-8 -*-
""" 
Sound functions for scikit-maad
Load, filter and transform 1D audio signals

"""

from .sound import (load,
                    envelope,
                    intoOctave,
                    audio_SNR,
                    spectral_SNR,
                    select_bandwidth,
                    fir_filter,
                    spectrogram,
                    avg_power_spectro,
                    avg_amplitude_spectro)

__all__ = ['load',
           'envelope',
           'intoOctave',
           'audio_SNR',
           'spectral_SNR',
           'select_bandwidth',
           'fir_filter',
           'spectrogram',
           'avg_power_spectro',
           'avg_amplitude_spectro']
