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
from scipy.optimize import root

# Import internal modules
from maad.util import moments, power2dB
from IPython.display import display
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
    >>> from maad import sound
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> Sxx_power,_,_,_ = sound.spectrogram (s, fs)

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
    Length of sk is : 512

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
    >>> print('Peak Frequency: {:.5f}, Amplitud: {:.5f}'.format(peak_freq, peak_freq_amp))
    Peak Frequency: 6634.16016, Amplitud: 0.00013

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
    >>> s, fs = sound.load('../../data/spinetail.wav')

    Compute the q-th quantile of the power spectrum

    >>> qs = features.spectral_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95])
    >>> print("5%: {:.2f} / 25%: {:.2f} / 50%: {:.2f} / 75%: {:.2f} \
    >>> / 95%: {:.2f}".format(qs[0], qs[1], qs[2], qs[3], qs[4]))
    5%: 6029.30 / 25%: 6416.89 / 50%: 6632.23 / 75%: 6890.62 / 95%: 9216.21

    Compute the q-th quantile of the power spectrum and its amplitude

    >>> qs = features.spectral_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95], 
    >>>                                 amp=True, as_pandas=True)
    >>> print(qs)
    index freq amp
    0.25  6416.894531  0.000092 / 0.50  6632.226562  0.000123
    0.75  6890.625000  0.000029 / 0.95  9216.210938  0.000005

    """
    q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 1]")

    # Compute spectrum
    if roi is None:
        pxx, fidx = sound.spectrum(s, fs, nperseg, **kwargs)
    else:
        pxx, fidx = sound.spectrum(s, fs, nperseg,
                                   tlims=[roi.min_t, roi.max_t],
                                   flims=[roi.min_f, roi.max_f],
                                   **kwargs)
    pxx = pd.Series(pxx, index=fidx)

    # Compute spectral q
    norm_cumsum = pxx.cumsum()/pxx.sum()
    spec_quantile = []
    for quantile in q:
        spec_quantile.append(pxx.index[np.where(norm_cumsum>=quantile)[0][0]])

    if amp:
        if as_pandas:
            out = pd.DataFrame({"freq":spec_quantile, "amp":pxx[spec_quantile].values}, index=q)
        else:
            out = np.transpose(np.matrix([q, spec_quantile, pxx[spec_quantile]]))
    else:
        if as_pandas:
            out = pd.Series(spec_quantile, index=q)
        else:
            out = np.array(spec_quantile)

    return out

#%%
def temporal_quantile(s, fs, q=[0.05, 0.25, 0.5, 0.75, 0.95], nperseg=1024, roi=None, mode="spectrum",
                       Nt=32, env_mode="fast", as_pandas=False, amp=False, **kwargs):
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
        Length of segment to compute the FFT. The default is 1024.
    roi : pandas.Series, optional
        Region of interest where peak frequency will be computed.
        Series must have a valid input format with index: min_t, min_f, max_t, max_f.
        The default is None.
    mode : str, optional, default is 'spectrum'
        - 'spectrum' : The quantile is calculated in the espectrum.
        - 'amplitude' : The quantile is calculated in the sound wave.
    Nt : integer, optional, default is `32`
        Size of each frame. The largest, the highest is the approximation.
    env_mode : str, optional, default is `fast`
        - `fast` : The sound is first divided into frames (2d) using the
            function wave2timeframes(s), then the max of each frame gives a
            good approximation of the envelope.
        - `Hilbert` : estimation of the envelope from the Hilbert transform.
            The method is slow
    as_pandas: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
    amp : bool, default is False
        Return the quantiles with its amplitude. 
    Returns
    -------
    Pandas Series/DataFrame or Numpy array/Matrix
        Temporal quantiles of waveform and its amplitude (optional).

    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../../data/spinetail.wav')

    Compute the q-th temporal quantile in the spectrum

    >>> qt = features.temporal_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95])
    >>> print("5%: {:.4f} / 25%: {:.4f} / 50%: {:.4f} / 75%: {:.4f} \
    >>> / 95%: {:.4f}".format(qt[0], qt[1], qt[2], qt[3], qt[4]))
    5%: 1.2190 / 25%: 5.7121 / 50%: 11.8190 / 75%: 16.5558 / 95%: 17.7517

    Compute the q-th temporal quantile in the waveform, amplitude

    >>> qt = features.temporal_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95], \
    >>>                                 env_mode="fast", mode="amplitude", Nt=32)
    >>> print("5%: {:.2f} / 25%: {:.2f} / 50%: {:.2f} / 75%: {:.2f} \
    >>> / 95%: {:.2f}".format(qt[0], qt[1], qt[2], qt[3], qt[4]))
    5%: 1.2154 / 25%: 5.7071 / 50%: 11.8169 / 75%: 16.3564 / 95%: 17.7605

    """
    q = np.asanyarray(q)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 1]")

    # compute quantiles in the time amp
    if mode=="amplitude":
        if roi is None:
            min_t = 0
        else:
            s = sound.trim(s, fs, min_t=roi.min_t, max_t=roi.max_t)
            min_t = roi.min_t

        env = sound.envelope(s, env_mode, Nt)
        t = min_t+np.arange(0,len(env),1)*len(s)/fs/len(env)
        energy = pd.Series(env**2, index=t)

        # Compute temporal quantile
        norm_cumsum = energy.cumsum()/energy.sum()
        spec_quantile = []
        for quantile in q:
            spec_quantile.append(energy.index[np.where(norm_cumsum>=quantile)[0][0]])

        if amp:
            if as_pandas:
                out = pd.DataFrame({"time":spec_quantile, "amp":energy[spec_quantile].values}, index=q)
            else:
                out = np.transpose(np.matrix([q, spec_quantile, energy[spec_quantile]]))
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
                out = np.transpose(np.matrix([q, spec_quantile, Sxx[spec_quantile]]))
        else:
            if as_pandas:
                out = pd.Series(spec_quantile, index=q)
            else:
                out = np.array(spec_quantile)

        return out
    else:
        raise Exception("Invalid mode. Mode should be 'spectrum' or 'amplitude'")

#%%
def bandwidth(s, fs, nperseg=1024, roi=None, mode="quantile", method='fast',
              reference="peak", dB=3, as_pandas=False, amp=False, **kwargs):
    """
    Compute the bandwith of the power spectrum. If a region of interest with time
    and spectral limits is provided, the bandwidth is computed on the selection.

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
        Mode used to compute the spectral bandwidth.
        The default is 'quantile'.
    method : {'fast', 'best'}, optional
        Method used to compute the peak frequency.
        The default is 'fast'.
    reference : {'peak', 'central'}, optional
        Reference point to calculate the bandwith.
        Default is "peak"
    dB : float, optional
        The X dB bandwidth, the frequency at which the signal amplitude
        reduces by x dB (when x= 3 the frequency becomes half its value).
        Default is -3.
    as_pandas : bool
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
    bandwidth_xdB : float
        Bandwidth x-dB of the audio
    Examples
    --------
    >>> from maad import features, sound
    >>> s, fs = sound.load('../../data/spinetail.wav')

    Compute the bandwidth of the power spectrum using quantiles

    >>> bw, bw_90 = features.bandwidth(s, fs, mode="quantile")
    >>> print("Bandwidth 50% : {:.4f} / Bandwidth 90% : {:.4f}".format(bw, bw_90))
    Bandwidth 50% : 473.7305 / Bandwidth 90% : 3186.9141

    Compute the 3-dB bandwidth of the power spectrum using the peak frequency as reference

    >>> bw_xdB = features.bandwidth(s, fs, mode="3dB", reference="peak", dB=3, as_pandas=True)
    >>> print("{} : {:.4f} ".format(bw_xdB.index[0], bw_xdB.values[0]))
    bandwidth_3dB : 320.8791 

    """
    # Compute spectral bandwith with the quantiles
    if mode=="quantile":
        q = spectral_quantile(s, fs, [0.05, 0.25, 0.75, 0.95], 
                              nperseg, roi, False, True, **kwargs)
        # Compute spectral bandwidth
        if amp:
            if as_pandas:
                out = pd.Series({"freq":[q[2,1]-q[1,1], q[3,1]-q[0,1]],
                                 "amp":[q[2,2], q[0,2]]},
                                index=["bandwidth_50", "bandwidth_90"])
            else:
                out = np.matrix([[np.abs(q[2,1]-q[1,1]), q[2,2]], [np.abs(q[3,1]-q[0,1]). q[0,2]]])
        else:
            if as_pandas:
                out = pd.Series([np.abs(q[2,1]-q[1,1]), np.abs(q[3,1]-q[0,1])],
                                index=["bandwidth_50", "bandwidth_90"])
            else:
                out = np.array([np.abs(q[2,1]-q[1,1]), np.abs(q[3,1]-q[0,1])])

        return out
    # Compute spectral X-bB bandwidth
    if mode=="3dB":
        if roi is None:
            pxx, fidx = sound.spectrum(s, fs, nperseg, **kwargs)
        else:
            pxx, fidx = sound.spectrum(s, fs, nperseg,
                                       tlims=[roi.min_t, roi.max_t],
                                       flims=[roi.min_f, roi.max_f],
                                       **kwargs)
        pxx_dB = pd.Series(power2dB(pxx), index=fidx)

        # compute peak frequency and its amplitud in dB
        if reference=="peak":
            _, amp_peak_freq = peak_frequency(s, fs, method, nperseg, roi, True, False)
            amp_peak_freq = power2dB(amp_peak_freq)
            threshold_array = (amp_peak_freq-dB)*np.ones_like(pxx_dB)
        # compute central frequency and its amplitud in dB
        elif reference=="central":
            central_freq_matrix = spectral_quantile(s, fs, [0.5], nperseg, roi, False, True)
            central_freq_amp = power2dB(central_freq_matrix[0,2])
            threshold_array = (central_freq_amp-dB)*np.ones_like(pxx_dB)
        else:
            raise Exception("Invalid reference. Reference should be 'central' or 'peak'")
    
        pxx_dB_shifted = pxx_dB.values - threshold_array

        index = np.where(np.diff(np.sign(pxx_dB_shifted)))[0]
        x,y = [], []
        for i in index:
            ind = np.argmin([pxx_dB_shifted[i]*pxx_dB_shifted[i-1],
                            pxx_dB_shifted[i]*pxx_dB_shifted[i+1]])
            ind1, ind2 = i, i-1+2*ind
            xi = fidx[np.sort([ind1, ind2])]

            f = interpolate.interp1d([fidx[ind1], fidx[ind2]], 
                                    [pxx_dB_shifted[ind1], pxx_dB_shifted[ind2]], 
                                    kind='linear')
            x0 = root(f, xi[0])["x"][0]
            x.append(x0); y.append(float(f(x0)))

        if amp:
            if as_pandas:
                    out = pd.DataFrame({"freq":x[-1]-x[0], "amp":y[-1]+threshold_array[0]},
                                        index=["bandwidth_{}dB".format(dB)])
            else:   out = np.array([x[-1]-x[0], y[-1]+threshold_array[0]])
        else:
            if as_pandas:
                    out = pd.Series(x[-1]-x[0], index=["bandwidth_{}dB".format(dB)])
            else:   out = x[-1]-x[0]

        return out
    else:
        raise Exception("Invalid mode. Mode should be 'quantile' or '3dB'")

#%%
def all_spectral_features(s, fs, nperseg=1024, roi=None, method='fast', dB=3, display=False, **kwargs):
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
    dB : float, optional
        The X dB bandwidth, the frequency at which the signal amplitude
        reduces by x dB (when x= 3 the frequency becomes half its value).
        Default is -3.
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
    >>> s, fs = sound.load('../../data/spinetail.wav')

    Compute all the spectral features

    >>> all_spectral_features = features.all_spectral_features(s, fs)
         sm            sv      ...   Time 75%   Time 95%  bandwidth_50  bandwidth_90  bandwidth_3dB
    0  0.000002  8.118042e-11  ...  16.555828  17.751655   473.730469    3186.914062    320.8791
    """

    Sxx_power,_,_,_ = sound.spectrogram (s, fs, window='hann', nperseg=nperseg)
    S_power = sound.avg_power_spectro(Sxx_power)
    sm = spectral_moments(S_power)

    qs = spectral_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95], nperseg, roi, **kwargs)
    qt = temporal_quantile(s, fs, [0.05, 0.25, 0.5, 0.75, 0.95], nperseg, roi, mode="spectrum", **kwargs)
    bw_50, bw_90 = bandwidth(s, fs, nperseg, roi, 'quantile', method, "peak", **kwargs)
    peak_freq = peak_frequency(s, fs, method, nperseg, roi, **kwargs)
    bw_3dB = bandwidth(s, fs, nperseg, roi, '3dB', method, "peak", dB, **kwargs)

    spectral_features = pd.DataFrame({"sm":sm[0], "sv":sm[1], "ss":sm[2], "sk":sm[3],
                                      "Freq 5%":qs[0], "Freq 25%":qs[1], "Freq 50%":qs[2], 
                                      "Freq 75%":qs[3], "Freq 95%":qs[4], "peak_freq":peak_freq,
                                      "Time 5%":qt[0], "Time 25%":qt[1], "Time 50%":qt[2], 
                                      "Time 75%":qt[3], "Time 95%":qt[4],
                                      "bandwidth_50":bw_50, "bandwidth_90":bw_90,
                                      "bandwidth_3dB":bw_3dB}, index=[0])

    if display: print(spectral_features)

    return spectral_features