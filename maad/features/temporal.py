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
import pandas as pd
from numpy.lib.function_base import _quantile_is_valid
from scipy import interpolate

# Import internal modules
from maad.util import moments
from maad.sound import envelope, trim

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

    References
    ----------
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


#%%
def temporal_quantile(s, fs, q=[0.05, 0.25, 0.5, 0.75, 0.95], roi=None, mode="fast", Nt=32, as_pandas_series=False):
    """
    Compute the q-th temporal quantile of the waveform. If a region of interest with time and spectral limits is provided,
    the q-th temporal quantile is computed on the selection.

    Parameters
    ----------
    s : 1D array
        Input audio signal
    fs : float
        Sampling frequency of audio signal
    q : array or float, optional
        Quantile or sequence of quantiles to compute, which must be between 0 and 1
        inclusive.
        The defaul is [0.05, 0.25, 0.5, 0.75, 0.95].
    roi : pandas.Series, optional
        Region of interest where peak frequency will be computed. 
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    mode : str, optional, default is `fast`
        - `fast` : The sound is first divided into frames (2d) using the 
            function wave2timeframes(s), then the max of each frame gives a 
            good approximation of the envelope.
        - `Hilbert` : estimation of the envelope from the Hilbert transform. 
            The method is slow
    Nt : integer, optional, default is `32`
        Size of each frame. The largest, the highest is the approximation.
    as_pandas_series: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
    Returns
    -------
    Pandas Series or Numpy array
        Temporal quantiles of waveform. 

    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../../data/spinetail.wav')

    Compute the q-th temporal quantile of the wave energy

    >>> qt = features.temporal_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95])
    >>> print("5%: {:.2f} / 25%: {:.2f} / 50%: {:.2f} / 75%: {:.2f} / 95%: {:.2f}".format(qt[0], qt[1], qt[2], qt[3], qt[4]))
    5%: 1.22 / 25%: 5.71 / 50%: 11.82 / 75%: 16.36 / 95%: 17.76

    """
    q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    if roi is None:
        min_t = 0
    else:
        s = trim(s, fs, min_t = roi.min_t, max_t = roi.max_t)
        min_t = roi.min_t

    env = envelope(s, mode, Nt)
    t = min_t+np.arange(0,len(env),1)*len(s)/fs/len(env)
    energy = pd.Series(env**2, index=t)

    # Compute temporal quantile
    norm_cumsum = energy.cumsum()/energy.sum()
    spec_quantile = list()
    for quantile in q:
        spec_quantile.append(energy.index[np.where(norm_cumsum>=quantile)[0][0]])
    
    if as_pandas_series:  return pd.Series(spec_quantile, index=[qth for qth in q])
    else:                 return np.array(spec_quantile)
    
#%%
def temporal_duration(s, fs, roi=None, mode="fast", Nt=32, as_pandas_series=False):
    """ 
    Compute the temporal duration of the waveform. If a region of interest with time and spectral limits is provided,
    the temporal duration is computed on the selection.

    Parameters
    ----------
    s : 1D array
        Input audio signal
    fs : float
        Sampling frequency of audio signal
    roi : pandas.Series, optional
        Region of interest where peak frequency will be computed. 
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    mode : str, optional, default is `fast`
        - `fast` : The sound is first divided into frames (2d) using the 
            function wave2timeframes(s), then the max of each frame gives a 
            good approximation of the envelope.
        - `Hilbert` : estimation of the envelope from the Hilbert transform. 
            The method is slow
    Nt : integer, optional, default is `32`
        Size of each frame. The largest, the highest is the approximation.
    as_pandas_series: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
    Returns
    -------
    duration : float 
        Duration 50% of the audio
    duration_90 : float 
        Duration 90%  of the audio
    
    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../../data/spinetail.wav')
    
    Compute the temporal duration of the time energy

    >>> duration, duration_90 = features.temporal_duration(s, fs)
    >>> print("Duration: {:.4f} / Duration 90% {:.4f}".format(duration, duration_90))
    Duration: 10.6493 / Duration 90% 16.5451

    """
    # Compute temporal quantile
    q = temporal_quantile(s, fs, [0.05, 0.25, 0.75, 0.95], roi, mode, Nt, as_pandas_series)
    
    # Compute temporal duration
    if as_pandas_series: 
        return pd.Series([np.abs(q.iloc[2]-q.iloc[1]), np.abs(q.iloc[3]-q.iloc[0])], index=["duration", "duration_90"])
    else:                
        return np.array([np.abs(q[2]-q[1]), np.abs(q[3]-q[0])])

#%%
def min_time(s, fs, method='best', roi=None, mode='fast', Nt=32, as_pandas_series=False):
    """
    Compute the minimum time for audio signal. The minimum time is the time of
    minimum energy. If a region of interest with time and spectral limits is provided,
    the minimum time is computed on the selection.
    
    Parameters
    ----------
    s : 1D array
        Input audio signal
    fs : float
        Sampling frequency of audio signal
    method : {'fast', 'best'}, optional
        Method used to compute the minimum time. 
        The default is 'best'.
    roi : pandas.Series, optional
        Region of interest where minimum time will be computed. 
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    mode : str, optional, default is `fast`
        - `fast` : The sound is first divided into frames (2d) using the 
            function wave2timeframes(s), then the max of each frame gives a 
            good approximation of the envelope.
        - `Hilbert` : estimation of the envelope from the Hilbert transform. 
            The method is slow
    Nt : integer, optional, default is `32`
        Size of each frame. The largest, the highest is the approximation.
    as_pandas_series: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
    
    Returns
    -------
    min_time : float
        Minimum time of audio segment.
    amp_min_time : float
        Amplitud of the minimum time of audio segment.

    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../../data/spinetail.wav')
    
    Compute peak time with the fast method 
    
    >>> min_time, amp_min_time = features.min_time(s, fs)
    >>> print('Minimum Time: {:.5f} / Minimum Time Amplitud: {:.5E}'.format(min_time, amp_min_time))
    Minimum Time: 3.78561 / Minimum Time Amplitud: 1.80306E-06

    """
    if roi is None:
        min_t = 0
    else:
        s = trim(s, fs, min_t = roi.min_t, max_t = roi.max_t)
        min_t = roi.min_t
    
    # compute envelope, time, and energy
    env = envelope(s, mode, Nt)
    t = min_t+np.arange(0,len(env),1)*len(s)/fs/len(env)
    energy = pd.Series(env**2, index=t)
        
    # simplest form to get peak frequency, but less precise
    if method=='fast':
        min_time = energy.idxmin() 
        amp_min_time = energy[min_time]
    
    elif method=='best':        
        # use interpolation get more precise peak frequency estimation
        idxmin = energy.index.get_loc(energy.idxmin())
        x = energy.iloc[[idxmin-1, idxmin, idxmin+1]].index.values
        y = energy.iloc[[idxmin-1, idxmin, idxmin+1]].values
        f = interpolate.interp1d(x,y, kind='quadratic')
        xnew = np.arange(x[0], x[2], 1)
        ynew = f(xnew) 
        min_time = xnew[ynew.argmax()]
        amp_min_time = ynew[ynew.argmax()]

    else:
        raise Exception("Invalid method. Method should be 'fast' or 'best' ")
    
    if as_pandas_series:    return pd.Series([amp_min_time], index=[min_time])
    else:                   return np.array([min_time, amp_min_time])

#%%
def peak_time(s, fs, method='best', roi=None, mode='fast', Nt=32, as_pandas_series=False):
    """
    Compute the peak time for audio signal. The peak time is the time of
    maximum energy. If a region of interest with time and spectral limits is provided,
    the peak time is computed on the selection.
    
    Parameters
    ----------
    s : 1D array
        Input audio signal
    fs : float
        Sampling frequency of audio signal
    method : {'fast', 'best'}, optional
        Method used to compute the peak time. 
        The default is 'best'.
    roi : pandas.Series, optional
        Region of interest where peak time will be computed. 
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    mode : str, optional, default is `fast`
        - `fast` : The sound is first divided into frames (2d) using the 
            function wave2timeframes(s), then the max of each frame gives a 
            good approximation of the envelope.
        - `Hilbert` : estimation of the envelope from the Hilbert transform. 
            The method is slow
    Nt : integer, optional, default is `32`
        Size of each frame. The largest, the highest is the approximation.
    as_pandas_series: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
    
    Returns
    -------
    peak_time : float
        Peak time of audio segment.
    amp_peak_time : float
        Amplitud of the peak time of audio segment.

    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../../data/spinetail.wav')
    
    Compute peak time with the fast method 
    
    >>> peak_time, peak_time_amp = features.peak_time(s, fs)
    >>> print('Peak Time: {:.5f} / Peak Time Amplitud: {:.5f}'.format(peak_time, peak_time_amp))
    Peak Time: 1.34967 / Peak Time Amplitud: 0.17552

    """
    if roi is None:
        min_t = 0
    else:
        s = trim(s, fs, min_t = roi.min_t, max_t = roi.max_t)
        min_t = roi.min_t

    # compute envelope, time, and energy
    env = envelope(s, mode, Nt)
    t = min_t+np.arange(0,len(env),1)*len(s)/fs/len(env)
    energy = pd.Series(env**2, index=t)
        
    # simplest form to get peak frequency, but less precise
    if method=='fast':
        peak_time = energy.idxmax() 
        amp_peak_time = energy[peak_time]
    
    elif method=='best':        
        # use interpolation get more precise peak frequency estimation
        idxmax = energy.index.get_loc(energy.idxmax())
        x = energy.iloc[[idxmax-1, idxmax, idxmax+1]].index.values
        y = energy.iloc[[idxmax-1, idxmax, idxmax+1]].values
        f = interpolate.interp1d(x,y, kind='quadratic')
        xnew = np.arange(x[0], x[2], 1)
        ynew = f(xnew) 
        peak_time = xnew[ynew.argmax()]
        amp_peak_time = ynew[ynew.argmax()]

    else:
        raise Exception("Invalid method. Method should be 'fast' or 'best' ")
    
    if as_pandas_series:    return pd.Series([amp_peak_time], index=[peak_time])
    else:                   return np.array([peak_time, amp_peak_time])