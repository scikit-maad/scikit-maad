#!/usr/bin/env python
""" 
Collection of functions to extract features from music
"""
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License

# =============================================================================
# Load the modules
# =============================================================================
# Import external modules
import numpy as np

# Import internal modules
from maad.util import moments

#%%
# =============================================================================
# public functions
# =============================================================================
#%%
def temporal_moments(s):
    """
    Computes the first 4th moments of an audio signal, mean, variance, skewness, kurtosis.
    
    Parameters
    ----------
    s : 1D array
        Audio to process
        
    Returns
    -------
    mean : float 
        mean of the audio
    var : float 
        variance  of the audio
    skew : float
        skewness of the audio
    kurt : float
        kurtosis of the audio
        
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> sm, sv, ss, sk = maad.features.temporal_moments (s)
    >>> print('mean: %2.2f / var: %2.5f / skewness: %2.4f / kurtosis: %2.2f' % (sm, sv, ss, sk))
    mean: -0.00 / var: 0.00117 / skewness: -0.0065 / kurtosis: 24.71
    
    """
    # force s to be ndarray
    s = np.asarray(s)
    
    return moments(s)

#%%
def zero_crossing_rate(s, fs):
    """
    Compute the zero crossing rate feature of an audio signal.
    
    The zero-crossing rate is the rate of sign-changes along a signal, i.e., 
    the rate at which the signal changes from positive to zero to negative or 
    from negative to zero to positive. This feature has been used widely 
    in speech recognition and music information retrieval, 
    being a key feature to classify percussive sounds [1]_.
    
    Parameters
    ----------
    s : 1D array
        Audio to process (wav)
    fs : float
        Sampling frequency of the audio (Hz)

    Returns
    ------- 
    zcr : float   
        number of zero crossing /s

    Reference
    ---------
    .. [1] https://en.wikipedia.org/wiki/Zero-crossing_rate

    Examples
    --------
    >>> from maad import sound, features
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> features.zero_crossing_rate(s,fs)
    10500.397192384766    
    
    """
    zero_crosses = np.nonzero(np.diff(s > 0))[0]
    duration = len(s) / fs
    zcr = 1/duration * len(zero_crosses)
    
    return zcr




    