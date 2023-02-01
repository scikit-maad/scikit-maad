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
from scipy.signal import find_peaks

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
def temporal_quantile(s, fs, q=[0.05, 0.25, 0.5, 0.75, 0.95], roi=None, mode="fast", Nt=32,
                      as_pandas=False, amp=False):
    """
    Compute the q-th temporal quantile of the waveform. If a region of interest with time
    and spectral limits is provided, the q-th temporal quantile is computed on the selection.

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
    >>> print("5%: {:.2f} / 25%: {:.2f} / 50%: {:.2f} / 75%: {:.2f} \
    >>> / 95%: {:.2f}".format(qt[0], qt[1], qt[2], qt[3], qt[4]))
    5%: 1.22 / 25%: 5.71 / 50%: 11.82 / 75%: 16.36 / 95%: 17.76

    """
    q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    if roi is None:
        min_t = 0
    else:
        s = trim(s, fs, min_t=roi.min_t, max_t=roi.max_t)
        min_t = roi.min_t

    env = envelope(s, mode, Nt)
    t = min_t+np.arange(0,len(env),1)*len(s)/fs/len(env)
    energy = pd.Series(env**2, index=t)

    # Compute temporal quantile
    norm_cumsum = energy.cumsum()/energy.sum()
    spec_quantile = []
    for quantile in q:
        spec_quantile.append(energy.index[np.where(norm_cumsum>=quantile)[0][0]])

    if amp:
        if as_pandas:
            out = pd.DataFrame({"freq":spec_quantile, "amp":energy[spec_quantile].values}, index=q)
        else:
            out = np.transpose(np.matrix([q, spec_quantile, energy[spec_quantile]]))
    else:
        if as_pandas:
            out = pd.Series(spec_quantile, index=q)
        else:
            out = np.transpose(np.matrix([q, spec_quantile]))

    return out
#%%
def temporal_duration(s, fs, roi=None, mode="fast", Nt=32, as_pandas_series=False):
    """
    Compute the temporal duration of the waveform. If a region of interest with time
    and spectral limits is provided, the temporal duration is computed on the selection.

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
    >>> s, fs = sound.load('data/spinetail.wav')

    Compute the temporal duration of the time energy

    >>> duration, duration_90 = features.temporal_duration(s, fs)
    >>> print("Duration: {:.4f} / Duration 90% {:.4f}".format(duration, duration_90))
    Duration: 10.6493 / Duration 90% 16.5451

    """
    # Compute temporal quantile
    q = temporal_quantile(s, fs, [0.05, 0.25, 0.75, 0.95], roi, mode, Nt, as_pandas_series)

    # Compute temporal duration
    if as_pandas_series:
        out = pd.Series([np.abs(q.iloc[2]-q.iloc[1]), np.abs(q.iloc[3]-q.iloc[0])],
                         index=["duration", "duration_90"])
    else:
        out = np.array([np.abs(q[2]-q[1]), np.abs(q[3]-q[0])])

    return out

#%%
def pulse_rate(s, fs, roi=None, dB=3, mode='fast', dmin=0.1, tol=1e-2,
               threshold=1.0, reference="peak", Nt=32, as_pandas_series=False, **kwargs):
    """
    Compute the bandwith of the power spectrum. If a region of interest with time and
    spectral limits is provided, the bandwidth is computed on the selection.

    Parameters
    ----------
    s : 1D array
        Input audio signal
    fs : float
        Sampling frequency of audio signal
    nperseg : int, optional
        Length of segment to compute the FFT. The default is 1024.
    roi : pandas.Series, optional
        Region of interest where peak frequency will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    mode : {'quantile', '3dB'}, optional
        Method used to compute the spectral bandwidth.
        The default is 'quantile'.
    as_pandas_series: bool, optional
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
    dB: float, optional
        The X dB bandwidth, the frequency at which the signal amplitude
        reduces by x dB (when x= 3 the frequency becomes half its value).
        Default is -3.
    threshold : float
        Threshold to caculate the bandwith from the reference frequency.
        Default is 0.5.
    reference : {'peak', 'central'}, optional
        Reference point to calculate the bandwith.
        Default is "peak"
    kwargs : additional keyword arguments
        If `window='hann'`, additional keyword arguments to pass to
        `sound.spectrum`.
    Returns
    -------
    bandwidth : float
        Bandwidth 50% of the audio
    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('data/spinetail.wav') #../../

    Compute the 3dB bandwidth of the power spectrum

    >>> bw_3dB= features.pulse_rate(s, fs)
    >>> print("3dB bandwidth: {:.4f} ".format())
    3 dB bandwidth:
    # sound.spectrogram <-> sound.spectrum()
    """
    if roi is None:
        min_t = 0
    else:
        s = trim(s, fs, min_t = roi.min_t, max_t = roi.max_t)
        min_t = roi.min_t

    env = envelope(s, mode, Nt)
    t = np.arange(0,len(env),1)*len(s)/fs/len(env)
    #t = np.arange(0,len(env),1)/fs
    s_dB = pd.Series(20*np.log10(np.abs(env)), index=t)

    if reference=="peak":
        t_ref, s_dB_ref = s_dB.idxmax(), s_dB[s_dB.idxmax()]
    elif reference=="center":
        center = temporal_quantile(s, fs, [0.5], roi, "fast", 32, False, True)
        t_ref, s_dB_ref = center[0,0], center[0,1]
    else:
        raise Exception("Invalid reference. Reference should be 'peak' or 'center'")

    threshold_array = (s_dB_ref-dB)*np.ones_like(env)

    #s_dB = np.clip(s_dB, np.nanmean(s_dB), 0)
    #s_dB_mean = np.nanmean( s_dB)
    peaks, properties = find_peaks(s_dB.values, height=threshold_array[0],
                                   prominence=10, distance=100, threshold=.1)
    intersect_points = s_dB.where(np.abs(s_dB.values-threshold_array[0]) < tol).dropna()

    # import matplotlib.pyplot as plt
    # plt.plot(t, s_dB.values,'o-', ms=2)
    # plt.plot(t_ref, s_dB_ref, 'kx')
    # plt.plot(t, threshold_array, 'k--')
    # plt.plot(t[peaks], s_dB[t[peaks]], 'x')
    # plt.plot(intersect_points.index, intersect_points.values, 'bx')
    # plt.show()

    # if as_pandas_series is True:
    #     return Sxx_dB[intersect_points]
    # else:
    #     return np.transpose(np.matrix([Sxx_dB[intersect_points].index, Sxx_dB[intersect_points].values]))
    #     #return np.array([Sxx_dB[intersect_points].index, Sxx_dB[intersect_points].values])

    #Sxx_amplitude,tn,fn,_ = maad.sound.spectrogram (w, fs, nperseg=1024, mode='amplitude')
    #S_amplitude_mean = np.mean(Sxx_amplitude, axis=1)