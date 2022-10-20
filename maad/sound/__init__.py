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
    load_url
    load_spectrogram
    write
    
Preprocess audio
-----------------
.. autosummary::
    :toctree: generated/

    fir_filter
    sinc
    smooth
    select_bandwidth
    pcen
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
    linear_to_octave
    envelope
    spectrum
    resample
    trim
    normalize
    gain

Metrics
-------
.. autosummary::
    :toctree: generated/
    
    temporal_snr
    spectral_snr
    sharpness

"""

from .input_output import (load,
                           load_url,
                           load_spectrogram,
                           write)

from .filter import (select_bandwidth,
                     fir_filter,
                     sinc,
                     smooth)

from .spectral_subtraction import (pcen,
                                   remove_background,
                                   remove_background_morpho,
                                   remove_background_along_axis,
                                   median_equalizer)

from .trim_func import wave2frames

from .transform import (envelope,
                        spectrum,
                        resample,
                        trim,
                        normalize,
                        gain)

from .spectro_func import (spectrogram,
                           avg_power_spectro,
                           avg_amplitude_spectro,
                           linear_to_octave)
                          
from .metrics import (temporal_snr,
                      spectral_snr,
                      sharpness)

__all__ = [
        # io.py
        'load',
        'load_url',
        'load_spectrogram',
        'write',
        # filter.py
        'select_bandwidth',
        'fir_filter',
        'sinc',
        'smooth',
        # spectral_subtraction.py
        'pcen',
        'remove_background', 
        'remove_background_morpho',
        'remove_background_along_axis',
        'median_equalizer',
        # trim_func.py
        'wave2frames',
        # transform.py
        'envelope',
        'spectrum',
        'resample',
        'trim',
        'normalize',
        'gain',
        # spectro_func.py
        'spectrogram',
        'avg_power_spectro',
        'avg_amplitude_spectro',
        'linear_to_octave',
        # metrics.py
        'temporal_snr',
        'spectral_snr',
        'sharpness']