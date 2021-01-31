# -*- coding: utf-8 -*-
""" 
Sound processing
================

The module ``sound`` is an ensemble of functions to load and preprocess audio signals.

Input and output
-----------------
.. autosummary::
    :toctree: generated/

    load
    loadSpectro

Preprocess audio
-----------------
.. autosummary::
    :toctree: generated/

    fir_filter
    sinc
    smooth
    select_bandwidth
    remove_background
    remove_background_morpho
    remove_background_along_axis
    median_equalizer
    wave2frames
   
Transform audio
---------------
.. autosummary::
    :toctree: generated/
    
    spectrogram
    avg_power_spectro
    avg_amplitude_spectro
    intoOctave
    envelope
    psd

Metrics
-------
.. autosummary::
    :toctree: generated/
    
    temporal_SNR
    spectral_SNR
    sharpness

"""

from .input_output import (load,
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

from .transform import (envelope,
                        psd)

from .spectro_func import (spectrogram,
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
        # envelope_func.py
        'envelope',
        'psd',
        # spectro_func.py
        'spectrogram',
        'avg_power_spectro',
        'avg_amplitude_spectro',
        'intoOctave',
        # metrics.py
        'audio_SNR',
        'spectral_SNR',
        'sharpness']