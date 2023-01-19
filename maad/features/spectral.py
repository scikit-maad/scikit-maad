#!/usr/bin/env python
""" 
Collection of functions to extract spectral features from audio signals
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
from numpy.lib.function_base import _quantile_is_valid
# Import internal modules
from maad.util import moments
from maad import sound

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

#%%
def min_frequency(s, fs, method='best', nperseg=1024, roi=None, as_pandas_series=False, **kwargs):
    """
    Compute the minimum frequency for audio signal. The minimum frequency is the frequency of
    minimum power. If a region of interest with time and spectral limits is provided,
    the minimum frequency is computed on the selection.
    
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
    as_pandas_series: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
    kwargs : additional keyword arguments
        If `window='hann'`, additional keyword arguments to pass to
        `sound.spectrum`.

    Returns
    -------
    min_freq : float
        Minimum frequency of audio segment.
    amp_min_freq : float
        Amplitud of the minimum frequency of audio segment.

    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../../data/spinetail.wav')
    
    Compute peak frequency with the fast method 
    
    >>> min_freq, min_freq_amp = features.peak_frequency(s, fs)
    >>> print('Minimum Frequency: {:.5f} / Minimum Frequency Amplitud: {:.5E}'.format(min_freq, min_freq_amp))
    Minimum Frequency: 6634.16016 / Minimum Frequency Amplitud: 1.29992E-04

    """
    if roi is None:
        pxx, fidx = sound.spectrum(s, fs, nperseg, **kwargs)
    else:
        pxx, fidx = sound.spectrum(s, fs, nperseg, 
                                   tlims=[roi.min_t, roi.max_t], 
                                   flims=[roi.min_f, roi.max_f], 
                                   **kwargs)
    pxx = pd.Series(pxx, index=fidx)
        
    # simplest form to get peak frequency, but less precise
    if method=='fast':
        min_freq = pxx.idxmin() 
        amp_min_freq = pxx[min_freq]
    
    elif method=='best':        
        # use interpolation get more precise peak frequency estimation
        idxmin = pxx.index.get_loc(pxx.idxmin())
        x = pxx.iloc[[idxmin-1, idxmin, idxmin+1]].index.values
        y = pxx.iloc[[idxmin-1, idxmin, idxmin+1]].values
        f = interpolate.interp1d(x,y, kind='quadratic')
        xnew = np.arange(x[0], x[2], 1)
        ynew = f(xnew) 
        min_freq = xnew[ynew.argmin()]
        amp_min_freq = ynew[ynew.argmin()]

    else:
        raise Exception("Invalid method. Method should be 'fast' or 'best' ")
    
    if as_pandas_series:    return pd.Series([amp_min_freq], index=[min_freq])
    else:                   return np.array([min_freq, amp_min_freq])

#%%
def peak_frequency(s, fs, method='best', nperseg=1024, roi=None, as_pandas_series=False, **kwargs):
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
    as_pandas_series: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
    kwargs : additional keyword arguments
        If `window='hann'`, additional keyword arguments to pass to
        `sound.spectrum`.

    Returns
    -------
    peak_freq : float
        Peak frequency of audio segment.
    amp_peak_freq : float
        Amplitud of the peak frequency of audio segment.

    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../../data/spinetail.wav')
    
    Compute peak frequency with the fast method 
    
    >>> peak_freq, peak_freq_amp = features.peak_frequency(s, fs)
    >>> print('Peak Frequency: {:.5f} / Peak Frequency Amplitud: {:.5f}'.format(peak_freq, peak_freq_amp))
    Peak Frequency: 6634.16016 / Peak Frequency Amplitud: 0.00013

    """
    if roi is None:
        pxx, fidx = sound.spectrum(s, fs, nperseg, **kwargs)
    else:
        pxx, fidx = sound.spectrum(s, fs, nperseg, 
                                   tlims=[roi.min_t, roi.max_t], 
                                   flims=[roi.min_f, roi.max_f],
                                   **kwargs)
    pxx = pd.Series(pxx, index=fidx)
        
    # simplest form to get peak frequency, but less precise
    if method=='fast':
        peak_freq = pxx.idxmax() 
        amp_peak_freq = pxx[peak_freq]
    
    elif method=='best':        
        # use interpolation get more precise peak frequency estimation
        idxmax = pxx.index.get_loc(pxx.idxmax())
        x = pxx.iloc[[idxmax-1, idxmax, idxmax+1]].index.values
        y = pxx.iloc[[idxmax-1, idxmax, idxmax+1]].values
        f = interpolate.interp1d(x,y, kind='quadratic')
        xnew = np.arange(x[0], x[2], 1)
        ynew = f(xnew) 
        peak_freq = xnew[ynew.argmax()]
        amp_peak_freq = ynew[ynew.argmax()]

    else:
        raise Exception("Invalid method. Method should be 'fast' or 'best' ")
    
    if as_pandas_series:    return pd.Series([amp_peak_freq], index=[peak_freq])
    else:                   return np.array([peak_freq, amp_peak_freq])

#%%
def spectral_quantile(s, fs, q=[0.05, 0.25, 0.5, 0.75, 0.95], nperseg=1024, roi=None, as_pandas_series=False, **kwargs):
    """
    Compute the q-th quantile of the power spectrum. If a region of interest with time and spectral limits is provided,
    the q-th quantile is computed on the selection.

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
    as_pandas_series: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
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
    >>> s, fs = sound.load('../../data/spinetail.wav')
    
    Compute the q-th quantile of the power spectrum

    >>> qs = features.spectral_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95])
    >>> print("5%: {:.2f} / 25%: {:.2f} / 50%: {:.2f} / 75%: {:.2f} / 95%: {:.2f}".format(qs[0], qs[1], qs[2], qs[3], qs[4]))
    5%: 6029.30 / 25%: 6416.89 / 50%: 6632.23 / 75%: 6890.62 / 95%: 9216.21

    """
    q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")
    
    # Compute spectrum
    if roi is None:
        pxx, fidx = sound.spectrum(s, fs, nperseg, **kwargs)
    else:
        pxx, fidx = sound.spectrum(s, fs, nperseg,
                                   tlims=[roi.min_t, roi.max_t], 
                                   flims=[roi.min_f, roi.max_f], **kwargs)
    pxx = pd.Series(pxx, index=fidx)

    # Compute spectral q
    norm_cumsum = pxx.cumsum()/pxx.sum()
    spec_quantile = list()
    for quantile in q:
        spec_quantile.append(pxx.index[np.where(norm_cumsum>=quantile)[0][0]])
    
    if as_pandas_series:  return pd.Series(spec_quantile, index=[qth for qth in q])
    else:                 return np.array(spec_quantile)
#%%
def bandwidth(s, fs, nperseg=1024, roi=None,  bw_method="quantile", as_pandas_series=False, **kwargs):
    """
    Compute the bandwith of the power spectrum. If a region of interest with time and spectral limits is provided,
    the bandwidth is computed on the selection.

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
    bw_method : {'quantile', '3dB'}, optional
        Method used to compute the spectral bandwidth. 
        The default is 'quantile'.
    as_pandas_series: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
    kwargs : additional keyword arguments
        If `window='hann'`, additional keyword arguments to pass to
        `sound.spectrum`.
    Returns
    -------
    bandwidth : float 
        Bandwidth 50% of the audio
    bandwidth_90 : float 
        Bandwidth 90%  of the audio
    
    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../../data/spinetail.wav')

    Compute the bandwidth of the power spectrum

    >>> bw, bw_90 = features.bandwidth(s, fs)
    >>> print("Bandwidth: {:.4f} / Bandwidth 90% {:.4f}".format(bw, bw_90))
    Bandwidth: 473.7305 / Bandwidth 90% 3186.9141

    """    
    # Compute spectral quantiles
    if bw_method=="quantile":
        q = spectral_quantile(s, fs, [0.05, 0.25, 0.75, 0.95], nperseg, roi, as_pandas_series, **kwargs)
    elif bw_method=="3dB":
        pass

    # Compute spectral bandwidth
    if as_pandas_series: 
        return pd.Series([np.abs(q.iloc[2]-q.iloc[1]), np.abs(q.iloc[3]-q.iloc[0])], index=["bandwidth", "bandwidth_90"])
    else:                
        return np.array([np.abs(q[2]-q[1]), np.abs(q[3]-q[0])])
#%% ---------------
# # Frequencia
# # - peak frequency      ok
# # - spectrum quantile   ok
# # - Bandwidth           ok (3dB missing method)
# # - max freq            ok
# # - min freq            ok
# # Tiempo
# # - quantile                 ok
# # - Center time              ok
# # - Duration                 ok
# # - Call rate / Pulse Rate     https://www.avisoft.com/tutorials/measuring-sound-parameters-from-the-spectrogram-automatically/
# # Spectro-temporal            timeR seewave
# # - pitch tracking


# info, sox 
# load