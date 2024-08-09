#!/usr/bin/env python
"""
Collection of functions to extract features from audio signals
License: New BSD License
"""

# =============================================================================
# Load the modules
# =============================================================================
# Import external modules
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import root

# Import internal modules
from maad.util import moments
from maad.sound import envelope, trim
from maad import sound

# define internal functions
def _quantile_is_valid(q):
    """ Check if quantile is valid
        function from older version of numpy than 2.0.0
    """
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                return False
    else:
        if not (np.all(0 <= q) and np.all(q <= 1)):
            return False
    return True


#%%
# =============================================================================
# public functions
# =============================================================================
#%%
def temporal_moments(s, fs=None, roi=None):
    """
    Computes the first 4th moments of an audio signal, mean, variance, skewness, kurtosis.

    Parameters
    ----------
    s : 1D array
        Audio to process
    fs : float, optional
        Sampling frequency of audio signal
        The default is None
    roi : pandas.Series, optional
        Region of interest where peak frequency will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.        

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
    >>> from maad import sound, features
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> sm, sv, ss, sk = features.temporal_moments (s, fs)
    >>> print('mean: %2.2f / var: %2.5f / skewness: %2.4f / kurtosis: %2.2f' % (sm, sv, ss, sk))
    mean: -0.00 / var: 0.00117 / skewness: -0.0065 / kurtosis: 24.71

    """
    # force s to be ndarray
    s = np.asarray(s)

    if (roi is not None):
        if (fs is not None) :
            s = trim(s, fs, min_t=roi.min_t, max_t=roi.max_t)
        else : 
            raise ValueError("If 'roi' is not None, 'fs' must be defined")     
        
    return moments(s)

#%%
def zero_crossing_rate(s, fs, roi=None):
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
        Sampling frequency of audio signal
    roi : pandas.Series, optional
        Region of interest where peak frequency will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.        

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

    if (roi is not None):
        s = trim(s, fs, min_t=roi.min_t, max_t=roi.max_t)

    zero_crosses = np.nonzero(np.diff(s > 0))[0]
    duration = len(s) / fs
    zcr = 1/duration * len(zero_crosses)

    return zcr

#%%
def temporal_quantile(s, fs, q=[0.05, 0.25, 0.5, 0.75, 0.95], nperseg=1024, roi=None, mode="spectrum",
                      env_mode="fast", as_pandas=False, amp=False, **kwargs):
    """
    Compute the q-th temporal quantile of the waveform or spectrum. If a
    region of interest with time and spectral limits is provided, the q-th
    temporal quantile is computed on the selection.

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
    nperseg : int, optional
        Length of segment to compute the FFT when mode is spectrum. The default is 1024.
        Size of each frame to compute the envelope. The largest, the highest is
        the approximation. The default is 5000.
    roi : pandas.Series, optional
        Region of interest where peak frequency will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    mode : str, optional, default is 'spectrum'
        - 'spectrum' : The quantile is calculated in the espectrum.
        - 'envelope' : The quantile is calculated in the sound wave.
    env_mode : str, optional, default is `fast`
        - `fast` : The sound is first divided into frames (2d) using the
            function wave2timeframes(s), then the max of each frame gives a
            good approximation of the envelope.
        - `Hilbert` : estimation of the envelope from the Hilbert transform.
            The method is slow
    as_pandas: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
    amp: bool, default is False
        Return the quantiles with its amplitude.

    Returns
    -------
    quantiles: pandas Series/DataFrame or Numpy array
        Temporal quantiles of waveform and its amplitude (optional).

    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../data/spinetail.wav')

    Compute the q-th temporal quantile in the spectrum

    >>> qt = features.temporal_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95], as_pandas=True)
    >>> print(qt)
    0.05     1.219048
    0.25     5.712109
    0.50    11.818957
    0.75    16.555828
    0.95    17.751655
    dtype: float64

    Compute the q-th temporal quantile in the waveform, using the envelope

    >>> qt = features.temporal_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95], mode="envelope", as_pandas=True)
    >>> print(qt)
    0.05     1.208300
    0.25     5.716188
    0.50    11.804161
    0.75    15.731135
    0.95    17.752714
    dtype: float64

    """
    q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 1]")

    # compute quantiles in the time amp
    if mode=="envelope":
        if roi is None:
            min_t = 0
        elif fs is not None:
            s = sound.trim(s, fs, min_t=roi.min_t, max_t=roi.max_t)
            min_t = roi.min_t
        else :
            raise ValueError("If 'roi' is not None, 'fs' must be defined")

        env = sound.envelope(s**2, env_mode, nperseg)
        t = min_t+np.arange(0,len(env),1)*len(s)/fs/len(env)
        energy = pd.Series(env, index=t)

        # Compute temporal quantile
        norm_cumsum = energy.cumsum()/energy.sum()
        spec_quantile = []
        for quantile in q:
            spec_quantile.append(energy.index[np.where(norm_cumsum>=quantile)[0][0]])

        if amp:
            if as_pandas:
                out = pd.DataFrame({"time":spec_quantile, "amp":energy[spec_quantile].values}, index=q)
            else:
                out = np.transpose(np.array([q, spec_quantile, energy[spec_quantile]]))
        else:
            if as_pandas:
                out = pd.Series(spec_quantile, index=q)
            else:
                out = np.array(spec_quantile)

        return out

    elif mode=="spectrum":
        if roi is None:
            Sxx,tn,_,_ = sound.spectrogram(s, fs, nperseg=nperseg, **kwargs)
        else:
            Sxx,tn,_,_ = sound.spectrogram(s, fs, nperseg=nperseg,
                                        tlims=[roi.min_t, roi.max_t],
                                        flims=[roi.min_f, roi.max_f],
                                        **kwargs)
        Sxx = pd.Series(np.average(Sxx,axis=0), index=tn)

        # Compute spectral q
        norm_cumsum = Sxx.cumsum()/Sxx.sum()
        spec_quantile = []
        for quantile in q:
            spec_quantile.append(Sxx.index[np.where(norm_cumsum>=quantile)[0][0]])

        if amp:
            if as_pandas:
                out = pd.DataFrame({"time":spec_quantile, "amp":Sxx[spec_quantile].values}, index=q)
            else:
                out = np.transpose(np.array([q, spec_quantile, Sxx[spec_quantile]]))
        else:
            if as_pandas:
                out = pd.Series(spec_quantile, index=q)
            else:
                out = np.array(spec_quantile)

        return out
    else:
        raise Exception("Invalid mode. Mode should be 'spectrum' or 'envelope'")
    
#%%
def temporal_duration(s, fs, nperseg=1024, roi=None, mode="spectrum",
                      env_mode="fast", as_pandas=False, **kwargs):
    """
    Compute the temporal duration of the waveform. If a region of interest with time
    and spectral limits is provided, the temporal duration is computed on the selection.

    Parameters
    ----------
    s : 1D array
        Input audio signal
    fs : float
        Sampling frequency of audio signal
    nperseg : int, optional
        Length of segment to compute the FFT. The default is 1024.
    roi : pandas.Series, optional
        Region of interest where temporal duration will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    mode : str, optional, default is 'spectrum'
        - 'spectrum' : The quantile is calculated using the spectrum.
        - 'amplitude' : The quantile is calculated using the enveloppe sound wave.
    env_mode : str, optional, default is `fast`
        - `fast` : The sound is first divided into frames (2d) using the
            function wave2timeframes(s), then the max of each frame gives a
            good approximation of the envelope.
        - `Hilbert` : estimation of the envelope from the Hilbert transform.
            The method is slow
    as_pandas_series: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.

    Returns
    -------
    duration: pandas Series/DataFrame or Numpy array
        Temporal duration of signal using energy quantiles.

    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../data/spinetail.wav')

    Compute the temporal duration of the time energy

    >>> duration, duration_90 = features.temporal_duration(s, fs)
    >>> print("Duration 50%: {:.4f} / Duration 90%: {:.4f}".format(duration, duration_90))
    Duration 50%: 10.8437 / Duration 90%: 16.5326

    """
    # Compute temporal quantile
    q = temporal_quantile(s, fs, [0.05, 0.25, 0.75, 0.95], nperseg, roi, mode, 
                          env_mode, as_pandas, **kwargs)

    # Compute temporal duration
    if as_pandas:
        out = pd.Series([np.abs(q.iloc[2]-q.iloc[1]), np.abs(q.iloc[3]-q.iloc[0])],
                        index=["duration_50", "duration_90"])
    else:
        out = np.array([np.abs(q[2]-q[1]), np.abs(q[3]-q[0])])

    return out

#%%
def all_temporal_features(s, fs, nperseg=1024, roi=None, display=False, **kwargs):
    """
    Compute all the temporal features for a signal.

    Parameters
    ----------
    s : 1D array
        Input audio signal
    fs : float
        Sampling frequency of audio signal
    nperseg : int, optional
        Length of segment to compute the FFT. The default is 1024.
    roi : pandas.Series, optional
        Region of interest where temporal features will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    kwargs : additional keyword arguments
        If `window='hann'`, additional keyword arguments to pass to
        `sound.spectrum`.

    Returns
    -------
    temporal_features : pandas DataFrame
        DataFrame with all temporal features computed in the spectrum
    
    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../data/spinetail.wav')

    Compute all the temporal features

    >>> temporal_features = features.all_temporal_features(s,fs)
    >>> print(temporal_features.iloc[0])
    sm            -2.043264e-19
    sv             1.167074e-03
    ss            -6.547980e-03
    sk             2.471161e+01
    Time 5%        1.219048e+00
    Time 25%       5.712109e+00
    Time 50%       1.181896e+01
    Time 75%       1.655583e+01
    Time 95%       1.775166e+01
    zcr            1.050040e+04
    duration_50    1.001495e+01
    duration_90    1.654441e+01
    Name: 0, dtype: float64
    """

    tm  = temporal_moments(s, fs, roi)
    zcr = zero_crossing_rate(s, fs, roi)
    qt = temporal_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95], nperseg, roi, mode="spectrum", **kwargs)
    duration_50, duration_90 = temporal_duration(s, fs, nperseg, roi, mode="envelope")

    temporal_features = pd.DataFrame({"sm":tm[0], "sv":tm[1], "ss":tm[2], "sk":tm[3],
                                    "Time 5%":qt[0], "Time 25%":qt[1], "Time 50%":qt[2], 
                                    "Time 75%":qt[3], "Time 95%":qt[4], "zcr":zcr,
                                    "duration_50":duration_50, "duration_90":duration_90}, index=[0])

    if display: print(temporal_features)

    return temporal_features

if __name__ == "__main__":
    import doctest
    doctest.testmod()