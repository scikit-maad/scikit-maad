#!/usr/bin/env python
"""
Collection of functions to extract spectral features from audio signals
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
from maad.util import moments, power2dB
from maad import sound

# define internal functions
def _interpolate_peak_location(pxx):
    """ Estimate peak using quadratic interpolation """
    idxmax = pxx.index.get_loc(pxx.idxmax())
    if (idxmax==0) | (idxmax+1==len(pxx)):
        # if maximum is at beginning or at the end os spectrum, take maximum
        peak = pxx.idxmax()
        amplitude = pxx[peak]
    else:
        # use interpolation
        x = pxx.iloc[[idxmax-1, idxmax, idxmax+1]].index.values
        y = pxx.iloc[[idxmax-1, idxmax, idxmax+1]].values
        f = interpolate.interp1d(x,y, kind='quadratic')
        xnew = np.arange(x[0], x[2], 1)
        ynew = f(xnew)
        peak = xnew[ynew.argmax()]
        amplitude = ynew[ynew.argmax()]
    return peak, amplitude

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
    >>> from maad import sound, features
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> Sxx_power,_,_,_ = sound.spectrogram(s, fs)

    Compute spectral moments on the mean spectrum

    >>> import numpy as np
    >>> S_power = sound.avg_power_spectro(Sxx_power)
    >>> sm, sv, ss, sk = features.spectral_moments (S_power)
    >>> print('mean: %2.8f / var: %2.10f / skewness: %2.2f / kurtosis: %2.2f' % (sm, sv, ss, sk))
    mean: 0.00000228 / var: 0.0000000001 / skewness: 5.84 / kurtosis: 40.49

    Compute spectral moments of the spectrogram along the time axis

    >>> from maad.features import spectral_moments
    >>> sm_per_bin, sv_per_bin, ss_per_bin, sk_per_bin = spectral_moments(Sxx_power, axis=1)
    >>> print('Length of sk_per_bin is : %2.0f' % len(sk_per_bin))
    Length of sk_per_bin is : 512

    """
    # force P to be ndarray
    X = np.asarray(X)

    return moments(X,axis)

#%%
def peak_frequency(s, fs, method='best', nperseg=1024, roi=None, amp=False, as_pandas=False, **kwargs):
    """
    Compute the peak frequency for audio signal. The peak frequency is the frequency of
    maximum power. If a region of interest with time and spectral limits is provided,
    the peak frequency is computed on the selection.

    Parameters
    ----------
    s : 1D array
        Input audio signal
    fs : float
        Sampling frequency of audio signal
    method : {'fast', 'best'}, optional
        Method used to compute the peak frequency.
        The default is 'best'.
    nperseg : int, optional
        Length of segment to compute the FFT. The default is 1024.
    roi : pandas.Series, optional
        Region of interest where peak frequency will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    as_pandas: bool
        Return data as a pandas.Series or pandas.DataFrame, when amp
        is False or True, respectively. Default is False.
    kwargs : additional keyword arguments
        Additional keyword arguments to pass to `sound.spectrum`.

    Returns
    -------
    peak_freq : float
        Peak frequency of audio segment.
    amp_peak_freq : float
        Amplitud of the peak frequency of audio segment.

    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../data/spinetail.wav')

    Compute peak frequency

    >>> peak_freq, peak_freq_amp = features.peak_frequency(s, fs, amp=True)
    >>> print('Peak Frequency: {:.5f}, Amplitude: {:.5f}'.format(peak_freq, peak_freq_amp))
    Peak Frequency: 6634.16016, Amplitude: 0.00013

    """
    if roi is None:
        pxx, fidx = sound.spectrum(s, fs, nperseg, **kwargs)
    else:
        pxx, fidx = sound.spectrum(s, fs, nperseg,
                                   tlims=[roi.min_t, roi.max_t],
                                   flims=[roi.min_f, roi.max_f],
                                   **kwargs)
    pxx = pd.Series(pxx, index=fidx)

    if method=='fast':
        # simplest form to get peak frequency, but less precise
        peak_freq = pxx.idxmax()
        amp_peak_freq = pxx[peak_freq]

    elif method=='best':
        # use interpolation get better estimate of peak frequency
        peak_freq, amp_peak_freq = _interpolate_peak_location(pxx)

    else:
        raise Exception("Invalid method. Method should be 'fast' or 'best' ")

    if amp:
        if as_pandas:
            out = pd.DataFrame({"freq":peak_freq, "amp":amp_peak_freq}, index=[0])
        else:
            out = np.array([peak_freq, amp_peak_freq])
    else:
        if as_pandas:
            out = pd.Series([peak_freq], index=["peak_freq"])
        else:
            out = peak_freq
    
    return out

#%%
def spectral_quantile(s, fs, q=[0.05, 0.25, 0.5, 0.75, 0.95], nperseg=1024, roi=None,
                      as_pandas=False, amp=False, **kwargs):
    """
    Compute the q-th quantile of the power spectrum. If a region of interest with time and
    spectral limits is provided, the q-th quantile is computed on the selection.

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
        Length of segment to compute the FFT. The default is 1024.
    roi : pandas.Series, optional
        Region of interest where peak frequency will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    as_pandas: bool
        Return data as a pandas.Series or pandas.DataFrame, when amp
        is False or True, respectively. Default is False.
    amp: bool
        Enable quantiles amplitude output. Default is False.
    kwargs : additional keyword arguments
        If `window='hann'`, additional keyword arguments to pass to
        `sound.spectrum`.
        
    Returns
    -------
    Pandas Series or Numpy array
        Quantiles of power spectrum.

    Examples
    --------
    >>> from maad import sound, features
    >>> s, fs = sound.load('../data/spinetail.wav')

    Compute the q-th quantile of the power spectrum

    >>> qs = features.spectral_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95], as_pandas=True)
    >>> print(qs)
    0.05    6029.296875
    0.25    6416.894531
    0.50    6632.226562
    0.75    6890.625000
    0.95    9216.210938
    dtype: float64

    Compute the q-th quantile of the power spectrum and its amplitude

    >>> qs = features.spectral_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95], amp=True, as_pandas=True)
    >>> print(qs)
                 freq       amp
    0.05  6029.296875  0.000007
    0.25  6416.894531  0.000067
    0.50  6632.226562  0.000087
    0.75  6890.625000  0.000022
    0.95  9216.210938  0.000004

    """
    q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 1]")

    # Compute spectrum
    # if roi is None:
    #     pxx, fidx = sound.spectrum(s, fs, nperseg, **kwargs)
    # else:
    #     pxx, fidx = sound.spectrum(s, fs, nperseg,
    #                                tlims=[roi.min_t, roi.max_t],
    #                                flims=[roi.min_f, roi.max_f],
    #                                **kwargs)
    # pxx = pd.Series(pxx, index=fidx)
    if roi is None:
        Sxx,_,fn,_ = sound.spectrogram(s, fs, nperseg=nperseg, **kwargs)
    else:
        Sxx,_,fn,_ = sound.spectrogram(s, fs, nperseg=nperseg,
                                       tlims=[roi.min_t, roi.max_t],
                                       flims=[roi.min_f, roi.max_f],
                                        **kwargs)
    pxx = pd.Series(np.average(Sxx,axis=1), index=fn)
    
    # Compute spectral q
    norm_cumsum = pxx.cumsum()/pxx.sum()
    spec_quantile = []
    for quantile in q:
        spec_quantile.append(pxx.index[np.where(norm_cumsum>=quantile)[0][0]])

    if amp:
        if as_pandas:
            out = pd.DataFrame({"freq":spec_quantile, "amp":pxx[spec_quantile].values}, index=q)
        else:
            out = np.transpose(np.array([q, spec_quantile, pxx[spec_quantile]]))
    else:
        if as_pandas:
            out = pd.Series(spec_quantile, index=q)
        else:
            out = np.array(spec_quantile)

    return out

#%%
def spectral_bandwidth(s, fs, nperseg=1024, roi=None, as_pandas=False, **kwargs):
    """
    Compute the bandwith of the power spectrum. If a region of interest with time
    and spectral limits is provided, the spectral bandwidth is computed on the selection.

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
    as_pandas : bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
    kwargs : additional keyword arguments
        If `window='hann'`, additional keyword arguments to pass to
        `sound.spectrum`.

    Returns
    -------
    bandwidth_50 : float
        Bandwidth 50% of the audio
    bandwidth_90 : float
        Bandwidth 90%  of the audio
    
    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> bw_50, bw_90 = features.spectral_bandwidth(s, fs, nperseg=1024)
    >>> print("Bandwidth 50% : {:.4f} / Bandwidth 90% : {:.4f}".format(bw_50, bw_90))
    Bandwidth 50% : 473.7305 / Bandwidth 90% : 3186.9141

    """
    # Compute spectral bandwith with the quantiles
    q = spectral_quantile(
        s, fs, [0.05, 0.25, 0.75, 0.95], nperseg, roi, False, True, **kwargs)
    
    out = np.array([np.abs(q[2,1]-q[1,1]), np.abs(q[3,1]-q[0,1])])

    if as_pandas:
        out = pd.Series(out, index=["bw_50", "bw_90"])
        
    return out

#%%
def all_spectral_features(s, fs, nperseg=1024, roi=None, method='fast', **kwargs):
    """
    Compute all the spectral features for a signal.

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
    method : {'fast', 'best'}, optional
        Method used to compute the peak frequency.
        The default is 'fast'.
    kwargs : additional keyword arguments
        If `window='hann'`, additional keyword arguments to pass to
        `sound.spectrum`.

    Returns
    -------
    spectral_features : pandas DataFrame
        DataFrame with all spectral features computed in the spectrum
    
    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../data/spinetail.wav')

    Compute all the spectral features

    >>> features.all_spectral_features(s, fs, nperseg=1024, roi=None)
    sm           2.276330e-06
    sv           8.118042e-11
    ss           5.844664e+00
    sk           4.048891e+01
    freq_05      6.029297e+03
    freq_25      6.416895e+03
    freq_50      6.632227e+03
    freq_75      6.890625e+03
    freq_95      9.216211e+03
    peak_freq    6.632227e+03
    bw_50        4.737305e+02
    bw_90        3.186914e+03
    dtype: float64
    """

    # Compute transformations
    Sxx_power,_,_,_ = sound.spectrogram (s, fs, window='hann', nperseg=nperseg)
    S_power = sound.avg_power_spectro(Sxx_power)
    
    # Compute features
    sm = spectral_moments(S_power)
    qs = spectral_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95], nperseg, roi, **kwargs)
    bw_50, bw_90 = spectral_bandwidth(s, fs, nperseg, roi, **kwargs)
    peak_freq = peak_frequency(s, fs, method, nperseg, roi, **kwargs)

    # Organize data into a Dataframe
    spectral_features = pd.Series({
        "sm":sm[0], 
        "sv":sm[1], 
        "ss":sm[2], 
        "sk":sm[3],
        "freq_05":qs[0], 
        "freq_25":qs[1], 
        "freq_50":qs[2], 
        "freq_75":qs[3], 
        "freq_95":qs[4], 
        "peak_freq":peak_freq,
        "bw_50":bw_50, 
        "bw_90":bw_90})

    return spectral_features

if __name__ == "__main__":
    import doctest
    doctest.testmod()