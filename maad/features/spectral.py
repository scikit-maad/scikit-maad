#!/usr/bin/env python
"""
Collection of functions to extract spectral features from audio signals
"""
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
# License: New BSD License

# =============================================================================
# Load the modules
# =============================================================================
# Import external modules
import numpy as np
import pandas as pd
from scipy import interpolate
from numpy.lib.function_base import _quantile_is_valid
# Import internal modules
from maad.util import moments
from maad import sound

# =============================================================================
# public functions
# =============================================================================
def spectral_moments (X, axis=None):
    """
    Computes the first 4th moments of an amplitude spectrum (1d) or spectrogram (2d),
    mean, variance, skewness, kurtosis.

    Parameters
    ----------
    X: ndarray of floats
        Amplitude  spectrum (1d) or spectrogram (2d).
    axis: interger, optional, default is None
        if spectrogram (2d), select the axis to estimate the moments.

    Returns
    -------
    mean: float
        mean of the audio
    var: float
        variance  of the audio
    skew: float
        skewness of the audio
    kurt: float
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
    >>> print('Length of sk_per_bin is: %2.0f' % len(sk_per_bin))
    Length of sk is: 512

    """
    # force P to be ndarray
    X = np.asarray(X)

    return moments(X,axis)

def peak_frequency(s, fs, nperseg=1024, roi=None, method='best'):
    """
    Compute the peak frequency for audio signal. The peak frequency is the frequency of
    maximum power. If a region of interest with time and spectral limits is provided,
    the peak frequency is computed on the selection.

    Parameters
    ----------
    s: 1D array
        Input audio signal
    fs: float
        Sampling frequency of audio signal
    nperseg: int, optional
        Length of segment to compute the FFT. The default is 1024.
    roi: pandas.Series, optional
        Region of interest where peak frequency will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    method: {'fast', 'best'}, optional
        Method used to compute the peak frequency.
        The default is 'fast'.

    Returns
    -------
    peak_freq: float
        Peak frequency of audio segment.

    """
    if roi is None:
        pxx, fidx = sound.spectrum(s, fs, nperseg)
    else:
        pxx, fidx = sound.spectrum(s, fs, nperseg,
                                   tlims=[roi.min_t, roi.max_t],
                                   flims=[roi.min_f, roi.max_f])
    pxx = pd.Series(pxx, index=fidx)

    # simplest form to get peak frequency, but less precise
    if method=='fast':
        peak_freq = pxx.idxmax()

    elif method=='best':
        # use interpolation get more precise peak frequency estimation
        idxmax = pxx.index.get_loc(pxx.idxmax())
        x = pxx.iloc[[idxmax-1, idxmax, idxmax+1]].index.values
        y = pxx.iloc[[idxmax-1, idxmax, idxmax+1]].values
        f = interpolate.interp1d(x,y, kind='quadratic')
        xnew = np.arange(x[0], x[2], 1)
        ynew = f(xnew)
        peak_freq = xnew[ynew.argmax()]

    else:
        raise Exception("Invalid method. Method should be 'fast' or 'best' ")

    return peak_freq

def spectral_quantile(s, fs, q, nperseg=1024, roi=None):
    """
    Compute the q-th quantile of the power spectrum.

    Parameters
    ----------
    s: 1D array
        Input audio signal
    fs: float
        Sampling frequency of audio signal
    q: array or float
        Quantile or sequence of quantiles to compute, which must be between 0 and 1
        inclusive.
    nperseg: int, optional
        Length of segment to compute the FFT. The default is 1024.
    roi : pandas.Series, optional
        Region of interest where peak frequency will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.

    Returns
    -------
    Pandas Series
        Quantiles of power spectrum.

    """
    q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    # Compute spectrum
    if roi is None:
        pxx, fidx = sound.spectrum(s, fs, nperseg)
    else:
        pxx, fidx = sound.spectrum(s, fs, nperseg,
                                   tlims=[roi.min_t, roi.max_t],
                                   flims=[roi.min_f, roi.max_f])
    pxx = pd.Series(pxx, index=fidx)

    # Compute spectral q
    norm_cumsum = pxx.cumsum()/pxx.sum()
    spec_quantile = list()
    for quantile in q:
        spec_quantile.append(pxx.index[np.where(norm_cumsum>=quantile)[0][0]])

    return pd.Series(spec_quantile, index=q)