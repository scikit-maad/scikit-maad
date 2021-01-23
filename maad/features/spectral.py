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
def spectral_moments (X, axis=None):
    """
    Computes the first 4th moments of an amplitude spectrum (1d) or spectrogram (2d), 
    mean, variance, skewness, kurtosis.
    
    Parameters
    ----------
    X : ndarray of floats
        Amplitude  spectrum (1d) or spectrogram (2d). 
    axis : interger, optional, default is None
        if spectrogram (2d), select the axis to estimate the moments.
        
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
    
    return moments(X,axis)

