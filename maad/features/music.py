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
from numpy import mean, var

# Import internal modules
from maad.util import kurtosis, skewness

#%%
# =============================================================================
# public functions
# =============================================================================
#%%
def moments (X, axis=None):
    """
    Computes the first 4th moments of a vector (1d, ie. spectrum or waveform) 
    or spectrogram (2d) 
    
    - mean
    - variance
    - skewness
    - kurtosis
    
    Parameters
    ----------
    X : ndarray of floats
        vector (1d : spectrum, waveform) or matrix (2d : spectrogram). 
    axis : interger, optional, default is None
        if spectrogram (2d), select the axis to estimate the moments.
        
    Returns
    -------
    mean : float 
        mean of X
    var : float 
        variance  of X
    skew : float
        skewness of X
    kurt : float
        kurtosis of X
        
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> Sxx_power,_,_,_ = maad.sound.spectrogram (s, fs) 
    
    Compute spectral moments on the mean spectrum
    
    >>> import numpy as np
    >>> S_power = maad.sound.avg_power_spectro(Sxx_power)
    >>> sm, sv, ss, sk = maad.features.spectral_moments (S_power)
    >>> print('mean: %2.8f / var: %2.10f / skewness: %2.2f / kurtosis: %2.2f' % (sm, sv, ss, sk))
    mean: 0.00000228 / var: 0.0000000001 / skewness: 5.84 / kurtosis: 40.49
    
    Compute spectral moments of the spectrogram along the time axis
    
    >>> sm_per_bin, sv_per_bin, ss_per_bin, sk_per_bin = maad.features.spectral_moments (Sxx_power, axis=1)
    >>> print('Length of sk_per_bin is : %2.0f' % len(sk_per_bin))
    Length of sk is : 512

    """
    # force P to be ndarray
    X = np.asarray(X)
    
    return mean(X, axis), var(X, axis), skewness(X, axis), kurtosis(X, axis)

#%%
def zero_crossing_rate(s, fs):
    """
    Compute the Zero Crossing Rate of an audio signal.
    
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

    Note
    ----
    From wikipedia :
    The zero-crossing rate is the rate of sign-changes along a signal, i.e., 
    the rate at which the signal changes from positive to zero to negative or 
    from negative to zero to positive.[1] This feature has been used heavily 
    in both speech recognition and music information retrieval, 
    being a key feature to classify percussive sounds.
    
    """
    zero_crosses = np.nonzero(np.diff(s > 0))[0]
    duration = len(s) / fs
    zcr = 1/duration * len(zero_crosses)
    
    return zcr




    