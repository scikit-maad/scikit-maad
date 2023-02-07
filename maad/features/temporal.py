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
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.optimize import root

# Import internal modules
from maad.util import moments
from maad.features import temporal_quantile
from maad.sound import envelope, trim

#%%
# =============================================================================
# public functions
# =============================================================================
#%%
def temporal_moments(s, fs, roi=None):
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
    >>> from maad import sound, features
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> sm, sv, ss, sk = features.temporal_moments (s, fs)
    >>> print('mean: %2.2f / var: %2.5f / skewness: %2.4f / kurtosis: %2.2f' % (sm, sv, ss, sk))
    mean: -0.00 / var: 0.00117 / skewness: -0.0065 / kurtosis: 24.71

    """
    # force s to be ndarray
    s = np.asarray(s)

    if roi is not None:
        s = trim(s, fs, min_t=roi.min_t, max_t=roi.max_t)

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
    >>> s, fs = sound.load('../../data/spinetail.wav')
    >>> features.zero_crossing_rate(s,fs)
    10500.397192384766

    """

    if roi is not None:
        s = trim(s, fs, min_t=roi.min_t, max_t=roi.max_t)

    zero_crosses = np.nonzero(np.diff(s > 0))[0]
    duration = len(s) / fs
    zcr = 1/duration * len(zero_crosses)

    return zcr

#%%
def temporal_duration(s, fs, nperseg=1024, roi=None, mode="spectrum",
                      env_mode="fast", Nt=32, as_pandas_series=False, **kwargs):
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
        Region of interest where peak frequency will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    mode : str, optional, default is 'spectrum'
        - 'spectrum' : The quantile is calculated in the espectrum.
        - 'amplitude' : The quantile is calculated in the sound wave.
    env_mode : str, optional, default is `fast`
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
    q = temporal_quantile(s, fs, [0.05, 0.25, 0.75, 0.95], nperseg, roi, mode, 
                          Nt, env_mode, as_pandas_series, **kwargs)

    # Compute temporal duration
    if as_pandas_series:
        out = pd.Series([np.abs(q.iloc[2]-q.iloc[1]), np.abs(q.iloc[3]-q.iloc[0])],
                         index=["duration", "duration_90"])
    else:
        out = np.array([np.abs(q[2]-q[1]), np.abs(q[3]-q[0])])

    return out

#%%
def pulse_rate(s, fs, roi=None, threshold1=3, threshold2=None, mode='fast', dmin=0.1, Nt=5000):
    """
    Find the events in the dB p 
    
    Parameters
    ----------
    s : 1D array
        Input audio signal
    fs : float
        Sampling frequency of audio signal
    roi : pandas.Series, optional
        Region of interest where peak frequency will be computed. Series must
        have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    threshold1 : float, optional
        The X dB threshold for events detections, the time at which 
        the signal amplitudereduces by x dB
        Default is 3.
    threshold2 : float, optional
        The X dB threshold for start/end events detections.
        Default is None.
    mode : str, optional, default is `fast`
        - `fast` : The sound is first divided into frames (2d) using the
            function wave2timeframes(s), then the max of each frame gives a
            good approximation of the envelope.
        - `Hilbert` : estimation of the envelope from the Hilbert transform.
            The method is slow
    dmin : float, optinal, default is 0.1
        Minimum time allowed for an event.
    Nt : integer, optional, default is `5000`
        Size of each frame. The largest, the highest is the approximation.
    Returns
    -------
    events : pandas DataFrame
        DataFrame with detected events: minimum time, maximum time and duration.
    amp : int
        Threshold amplitude for the event detection.
    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../../data/spinetail.wav')

    Detect events 15-dB below the peak with a minimum distance of 0.5 s.

    >>> t = np.arange(0,len(s),1)/fs
    >>> s_dB = 20*np.log10(np.abs(s))
    >>> events, amp = pulse_rate(s, fs, dmin=0.5, dB=20)

    Plot signal in dB and detected events, initial and final points

    >>> plt.figure()
    >>> plt.plot(t, s_dB)
    >>> plt.plot(events["min_t"], amp*np.ones_like(events["min_t"]), 'kx')
    >>> plt.plot(events["max_t"], amp*np.ones_like(events["max_t"]), 'kx')
    >>> plt.ylim((-50,0))
    >>> plt.show()

    """
    if roi is None:
        min_t = 0
    else:
        s = trim(s, fs, min_t=roi.min_t, max_t=roi.max_t)
        min_t = roi.min_t

    t = np.arange(0,len(s),1)/fs
    s_dB = 20*np.log10(np.abs(s))
    s_dB_mean = np.mean(s_dB)
    s_dB = np.clip(s_dB, s_dB_mean,0)-s_dB_mean

    env = envelope(s_dB, mode, Nt)
    t_env = np.arange(0,len(env),1)*len(s)/fs/len(env)

    s_dB_env = pd.Series(env, index=t_env)
    s_dB_ref = s_dB_env[s_dB_env.idxmax()]

    threshold_array = (s_dB_ref-threshold1)*np.ones_like(s_dB_env)

    s_dB_shifted = env-threshold_array
    index = np.where(np.diff(np.sign(s_dB_shifted)))[0]
    x,y = [], []
    for i in index:
        ind = np.argmin([s_dB_shifted[i]*s_dB_shifted[i-1],
                         s_dB_shifted[i]*s_dB_shifted[i+1]])
        ind1, ind2 = i, i-1+2*ind

        xi = t_env[np.sort([ind1, ind2])]

        f = interpolate.interp1d([t_env[ind1], t_env[ind2]], 
                                 [s_dB_shifted[ind1], s_dB_shifted[ind2]], 
                                 kind='linear')
        x0 = root(f, xi[0])["x"][0]
        x.append(x0); y.append(float(f(x0)))

    detected_events = [[x[i], x[i+1], x[i+1]-x[i]] for i in range(0,len(x),2)]
    events = np.array([x for x in detected_events if x[2]>dmin])

    events = pd.DataFrame({"min_t":events[:,0], "max_t":events[:,1], 
                           "duration":events[:,2]}, index=range(events.shape[0]))
    amp = s_dB_mean+threshold_array[0]

    if threshold2 is not None:
        events2 = []

    return events, amp

#%%
def all_temporal_features(s, fs, nperseg=1024, roi=None, threshold1=3, 
                          threshold2=None, dmin=0.1, Nt=5000, display=False):
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
        Region of interest where peak frequency will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    threshold1 : float, optional
        The X dB threshold for events detections, the time at which 
        the signal amplitudereduces by x dB
        Default is 3.
    threshold2 : float, optional
        The X dB threshold for start/end events detections.
        Default is None.
    dmin : float, optinal, default is 0.1
        Minimum time allowed for an event.
    Nt : integer, optional, default is `5000`
        Size of each frame. The largest, the highest is the approximation.
    kwargs : additional keyword arguments
        If `window='hann'`, additional keyword arguments to pass to
        `sound.spectrum`.
    Returns
    -------
    temporal_features : pandas DataFrame
        DataFrame with all spectral features computed in the spectrum
    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../../data/spinetail.wav')

    Compute all the temporal features

    >>> temporal_features = features.all_temporal_features(s,fs)
             sm        sv      ...   Time 75%   Time 95%       zcr     duration_50    duration_90
    0 -2.043264e-19  0.001167  ...  16.356414  17.760507  10500.397192    10.649338    16.545078
    """

    tm  = temporal_moments(s, fs, roi)
    zcr = zero_crossing_rate(s, fs, roi)
    qt  = temporal_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95], nperseg, roi, "amplitude")
    pr  = pulse_rate(s, fs, roi, threshold1, threshold2, 'fast', dmin, Nt)
    duration_50, duration_90 = temporal_duration(s, fs, nperseg, roi, mode="amplitude")

    temporal_features = pd.DataFrame({"sm":tm[0], "sv":tm[1], "ss":tm[2], "sk":tm[3],
                                      "Time 5%":qt[0], "Time 25%":qt[1], "Time 50%":qt[2], 
                                      "Time 75%":qt[3], "Time 95%":qt[4], "zcr":zcr,
                                      "duration_50":duration_50, "duration_90":duration_90}, index=[0])

    if display: print(temporal_features)

    return temporal_features