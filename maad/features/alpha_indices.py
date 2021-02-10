#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""  
Collection of functions to compute alpha acoustic indices to chracterise audio signals.
"""
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>        
#
# License: New BSD License

#%%
#***************************************************************************
# -------------------       Load modules         ---------------------------
#***************************************************************************
# Import external modules
import numbers
import numpy as np 
from numpy import sum, log, min, max, abs, mean, median, sqrt, diff, var
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.stats import rankdata
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd # for csv
# min value
import sys
_MIN_ = sys.float_info.min

# Import internal modules
from maad.util import (rle, index_bw, amplitude2dB, power2dB, dB2power, mean_dB,
                       skewness, kurtosis, format_features, into_bins, entropy, 
                       linear_scale, plot1d, plot2d, overlay_rois)
from maad.spl import wav2leq, psd2leq, power2dBSPL
from maad.features import (centroid_features, zero_crossing_rate, temporal_moments, 
                           spectral_moments)
from maad.sound import (envelope, smooth, temporal_snr, linear_to_octave, 
                        avg_amplitude_spectro, avg_power_spectro, spectral_snr, 
                        median_equalizer)
from maad.rois import select_rois, create_mask

#%%
# =============================================================================
# Private functions
# =============================================================================
def _acoustic_activity (xdB, dB_threshold, axis=1):
    """
    Acoustic Activity [1]_:
    
    for each frequency bin :
    - ACTfract : proportion (fraction) of points above the threshold 
    - ACTcount : number of points above the threshold
    - ACTmean : mean value (in dB) of the portion of the signal above the threhold
    
    Parameters
    ----------
    xdB : ndarray of floats
        1d : envelope in dB of the audio signal 
        2d : PSD spectrogram in dB
        It's better to work with PSD or envelope without background variation
        as the process is based on threshold.
    dB_threshold : scalar, optional, default is 6dB
        data >Threshold is considered to be an event 
        if the length is > rejectLength
    
    Returns
    -------    
    ACTfract :ndarray of scalars
        proportion (fraction) of points above the threshold for each frequency bin
    ACTcount: ndarray of scalars
        number of points above the threshold for each frequency bin
    ACTmean: scalar
        mean value (in dB) of the portion of the signal above the threhold
        
    References 
    ----------
    .. [1] Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived 
    from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    
    ACTsp [Towsey] : ACTfract (proportion (fraction) of point value above the theshold)
    EVNsp [Towsey] : ACTcount (number of point value above the theshold)
    """ 
    
    ### For x to be a ndarray
    xdB = np.asarray(xdB)
   
    ### compute _score
    ACTfract, ACTcount = _score(xdB, dB_threshold, axis=axis)
    ACTfract= ACTfract.tolist()
    ACTcount = ACTcount.tolist()
    ACTmean = mean_dB(xdB[xdB>dB_threshold])
    return ACTfract, ACTcount, ACTmean 

#%%    
def _acoustic_events(xdB, dt, dB_threshold=6, rejectDuration=None):
    """
    Acoustic events [1]_ :
        - EVNsum : total events duration (s) 
        - EVNmean : mean events duration (s)
        - EVNcount : number of events per s
    
    Parameters
    ----------
    xdB : ndarray of floats
        2d : Spectrogram  in dB

    dt : scalar
        Time resolution

    dB_threshold : scalar, optional, default is 6dB
        data >Threshold is considered to be an event 
        if the length is > rejectLength
        
    rejectDuration : scalar, optional, default is None
        event shorter than rejectDuration are discarded
        duration is in s
    
    Returns
    -------    
    EVNsum :scalar
        total events duration in s
    EVNmean: scalar
        mean events duration in s
    EVNcount: scalar
        number of events per s
    EVN: ndarray of floats 
        binary vector or matrix.
        1 corresponds to event
        0 corresponds to background

    References 
    ----------
    .. [1]  Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms 
            Derived from Natural Recordings of the Environment. 
            Queensland University of Technology, Brisbane.
    """    
    # total duration
    if xdB.ndim ==1 : duration = (len(xdB)-1) * dt
    if xdB.ndim ==2 : duration = (xdB.shape[1]-1) * dt
    
    xdB = np.asarray(xdB)
    # thresholding => binary
    EVN = (xdB>=dB_threshold)*1  
    # Remove events shorter than 'rejectLength' 
    # (done by erosion+dilation = opening)
    if rejectDuration is not None:
        rejectLength = int(round(rejectDuration / dt))
        # tricks. Depending on the dimension of bin_x 
        # if bin_x is a vector
        if EVN.ndim == 1 : kernel = np.ones(rejectLength+1)
        # if bin_x is a matrix
        elif EVN.ndim == 2 : kernel = [list(np.ones(rejectLength+1))]  
        else: print("xdB must be a vector or a matrix")
        # Morphological tool : Opening
        EVN = binary_erosion(EVN, structure=kernel)
        EVN = binary_dilation(EVN, structure=kernel) 
    
    # Extract the characteristics of each event : 
    # duration (mean and sum in s) and count
    if EVN.ndim == 2 :
        EVNsum = []
        EVNmean = []
        EVNcount = []
        for i, b in enumerate(EVN) :
            l, v = rle(b)  
            if sum(l[v==1])!=0 :
                # mean events duration in s
                EVNmean.append(mean(l[v==1]) * dt)
            else:
                EVNmean.append(0)    
            # total events duration in s 
            EVNsum.append(sum(l[v==1]) * dt)
            # number of events
            EVNcount.append(sum(v)/ duration)
    elif EVN.ndim == 1 :
        l, v = rle(EVN) 
        if sum(l[v==1]) !=0 :
            # mean events duration in s
            EVNmean = mean(l[v==1]) * dt
        else:
            EVNmean = 0
        # total events duration in s 
        EVNsum = sum(l[v==1]) * dt
        # number of events per s
        EVNcount = sum(v) / duration
    else: print("xdB must be a vector or a matrix")
    
    return EVNsum, EVNmean, EVNcount, EVN

#%%
def _score (x, threshold, axis=0):
    """
    Score

    count the number of times values in x that are greater than the threshold 
    and normalized by the total number of values in x
    
    Parameters
    ----------
    x : ndarray of floats
        Vector or matrix containing the data
        
    threshold : scalar
        Value > threshold are counted    
        
    axis : integer, optional, default is 0
        score is calculated along this axis.
        
    Returns
    -------    
    count : scalar
        the number of times values in x that are greater than the threshold
    s : scalar
        count is normalized by the total number of values in x
    """
    x = np.asarray(x)
    x = x>=threshold
    count = sum(x,axis=axis)
    s = sum(x,axis=axis)/x.shape[axis]
    return s, count

#%%
def _shannonEntropy(datain, axis=0):
    """
    Shannon Entropy
    
    Parameters
    ----------
    datain : ndarray of floats
        Vector or matrix containing the data
    
    axis : integer, optional, default is 0
        entropy is calculated along this axis.

    Returns
    -------    
    Hs : ndarray of floats
        Vector or matrix of Shannon Entropy
    """
    # length of datain along axis
    n = datain.shape[axis]
    Hs = entropy(datain, axis=axis) * np.log(n)
    return Hs

#%%
def _gini(x, corr=False):
    """
    Gini
    
    Compute the Gini value of x
    
    Parameters
    ----------
    x : ndarray of floats
        Vector or matrix containing the data
    
    corr : boolean, optional, default is False
        Correct the Gini value
        
    Returns
    -------  
    G: scalar
        Gini value
        
    References
    ----------
    Ported from ineq library in R
    """
    if sum(x) == 0:
       G = 0 # null gini
    else:
        n = len(x)
        x.sort()
        G = sum(x * np.arange(1,n+1,1))
        G = 2 * G/sum(x) - (n + 1)
        if corr : G = G/(n - 1)
        else : G= G/n
    return G

#%%
def _raoQ (p, bins):
    """
    Compute Rao's Quadratic entropy in 1d
    
    Parameters
    ---------
    p : ndarray of floats (1d)
        a vector containing the probality of each bin
    bins : ndarray of floats (1d)
        a vector containing the value of each bin
        
    Return
    ------
    Q : scalar
        Rao's Quadratic entropy value
    
    Reference:
    ---------
    .. [1] Botta-Dukát, Zoltán, Rao’s quadratic entropy as a measure of functional 
    diversity based on multiple traits, Journal of Vegetation Science, 2005
    
    """
    
    # be sure they are ndarray
    p = np.asarray(p)
    bins = np.asarray(bins)
    
    # Normalize p by the sum in order to get the sum of p = 1
    p = p/np.sum(p)
    
    # Bins is normalized by the bins range
    bins = bins/(bins.max() - bins.min())
    
    # take advantage of broadcasting, 
    # Get the pairwise distance 
    # Euclidian distance
    d = abs(bins[..., np.newaxis] - bins[np.newaxis, ...])
        
    # compute the crossproduct of pixels value pi,pj
    pipj = (p[..., np.newaxis] * p[np.newaxis, ...])

    # Multiply by 2*sqrt(2) to take into account the lower triangle (symmetric)
    Q = np.sum(np.sum(pipj*d))*2*sqrt(2)
    
    return Q

#%%
# =============================================================================
# Public functions
# =============================================================================
def surface_roughness (x, norm ='global'):
    
    """
    Compute the surface roughness index of a signal (1D) or a spectrogram (2D).
    
    Surface roughness is quantified by the deviations in the direction of the normal 
    vector of a real surface from its ideal form. If these deviations are large, 
    the surface is rough; if they are small, the surface is smooth [1]_.
    
    Parameters
    ----------
    x : ndarray of floats
        vector (1d) or matrix (2d)
    
    norm : string, optional, default is 'global'
        Determine if the ROUGHNESS is normalized by the sum of the whole data
        ('global' mode) or by the sum of horizontal line for each line
        ('per_bin')

    Returns
    -------                
    Ra : scalar or 1d ndarray of scalars
        if x is a vector => Arithmetical mean deviation of x.
        if x is a matrix => Arithmetical mean deviation of each line of x.
        
    Rq : scalar or 1d ndarray of scalars
        if x is a vector => Root mean squared of deviationn of x.
        if x is a matrix => Root mean squared of deviation of each line of x.
    
    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Surface_roughness
    
    """    
    # force to be ndarray
    x = np.asarray(x)
    
    if x.ndim == 1 :
        m = np.mean(x)
        y = x-m
        # Arithmetic mean deviation
        Ra = mean(abs(y))
        # Root mean square
        Rq = sqrt(mean(y**2))  
        
    elif x.ndim ==2 :
        if norm == 'per_bin':
            m = np.mean(x, axis=1)
            y = x-m[..., np.newaxis]
        elif norm == 'global':
            m = np.mean(x)
            y = x-m 
        else :
            raise TypeError ('norm has to be in {per_bin, global}')    
        
        # Arithmetic mean deviation
        Ra = mean(abs(y), axis=1)
        # Root mean square
        Rq = sqrt(mean(y**2, axis=1))   
        
    else :
        raise TypeError ('x should be a vector (1d) or a matrix (2d) of floats')
        
    return Ra, Rq

#=============================================================================
def roughness (x, norm=None, axis=0) :
    """
    Computes the roughness (depends on the number of peaks and their amplitude)
    of a vector or matrix x (i.e. waveform, spectrogram...)   
    Roughness = sum(second_derivation(x)²)
    
    Parameters
    ----------
    x : ndarray of floats
        x is a vector (1d) or a matrix (2d)
        
    norm : boolean, optional. Default is None
    
        - 'global' : normalize by the maximum value in the vector or matrix
        - 'per_axis' : normalize by the maximum value found along each axis

    axis : int, optional, default is 0
        select the axis where the second derivation is computed
        
        if x is a vector, axis=0
        
        if x is a 2d ndarray, axis=0 => rows, axis=1 => columns
                
    Returns
    -------
    y : float or ndarray of floats

    References
    ----------
    Described in [Ramsay JO, Silverman BW (2005) Functional data analysis.]
    Ported from SEEWAVE R Package
    """      
    
    if norm is not None:
        if norm == 'per_axis' :
            m = np.max(x, axis=axis) 
            m[m==0] = _MIN_    # Avoid dividing by zero value
            if axis==0:
                x = x/m[None,:]
            elif axis==1:
                x = x/m[:,None]
        elif norm == 'global' :
            m = np.max(x) 
            if m==0 : m = _MIN_    # Avoid dividing by zero value
            x = x/m 
            
    deriv2 = np.diff(x, 2, axis=axis)
    r = np.sum(deriv2**2, axis=axis)
    
    return r


#******************************************************************************
#               TEMPORAL ECOACOUSTICS INDICES
#******************************************************************************
#=============================================================================
def temporal_median (s, mode ='fast', Nt=512) :
    """
    Computes the median of the envelope of an audio signal.

    Parameters
    ----------
    s : 1D array
        Audio to process (wav)
    mode : str, optional, default is "fast"
        Select the mode to compute the envelope of the audio waveform
        - "fast" : The sound is first divided into frames (2d) using the 
            function _wave2timeframes(s), then the max of each frame gives a 
            good approximation of the envelope.
        - "Hilbert" : estimation of the envelope from the Hilbert transform. 
            The method is slow
    Nt : integer, optional, default is 512
        Size of each frame. The largest, the highest is the approximation.
    
    Returns
    -------
    MED: float
       Median of the envelope 

    Examples
    --------
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> med = maad.features.temporal_median(s)
    >>> print(med)
    0.007934564717486147
    
    """
    # Envelope
    env = envelope(s, mode=mode, Nt=Nt)
    # median
    MED = np.median(env)

    return MED

#=============================================================================
def temporal_entropy (s, compatibility="QUT", mode ='fast', Nt=512) :
    """
    Computes the entropy of the envelope of an audio signal.

    Parameters
    ----------
    s : 1D array
        Audio to process (wav)
    compatibility : string {'QUT', 'seewave'}, default is 'QUT'
        Select the way to compute the temporal entropy.
            - QUT [2]_: entropy of the squared envelope
            - seewave [1]_ : entropy of the envelope
    mode : str, optional, default is "fast"
        Select the mode to compute the envelope of the audio waveform.
            - "fast" : The sound is first divided into frames (2d) using the function _wave2timeframes(s), then the max of each frame gives a good approximation of the envelope.
            - "Hilbert" : estimation of the envelope from the Hilbert transform. The method is slow.
    Nt : integer, optional, default is 512
        Size of each frame. The largest, the highest is the approximation.
   
    Returns
    -------
    Ht: float
       Temporal entropy of the audio 
       
    References
    ----------
    .. [1] Seewave : http://rug.mnhn.fr/seewave/
    Sueur, J., Aubin, T., & Simonis, C. (2008). Seewave, a free modular tool 
    for sound analysis and synthesis. Bioacoustics, 18(2), 213-226.
     
    .. [2] QUT : https://github.com/QutEcoacoustics/audio-analysis/
    Michael Towsey, Anthony Truskinger, Mark Cottman-Fields, & Paul Roe. 
    (2018, March 5). Ecoacoustics Audio Analysis Software v18.03.0.41 (Version v18.03.0.41). 
    Zenodo. http://doi.org/10.5281/zenodo.1188744
    
    Notes
    -----
    The entropy of an audio signal is a measure of energy dispersion. In the temporal domain, 
    values below 0.7 indicate a brief concentration of energy (few miliseconds), while 
    values close 1 indicate low concentration of energy, no peaks, smooth and constant 
    background noise.

    Examples
    --------
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> Ht = maad.features.temporal_entropy (s)
    >>> print(Ht)
    0.7518917279549968

    """
    # Envelope
    env = envelope(s, mode=mode, Nt=Nt)
    # Entropy
    if compatibility == 'QUT':
        Ht = entropy(env**2)
    elif compatibility == 'seewave':
        Ht = entropy(env)
    else:
        raise TypeError('compatibility must be selected in {QUT, seewave}')  

    return Ht

#=============================================================================
def acoustic_richness_index (Ht_array, M_array):
    """
    Compute the acoustic richness index of an audio file. 
    
    This acoustic index was first described in [1]_. The present version was 
    translated from the R software package Seewave [2]_.
    
    Parameters
    ----------
    Ht_array : 1d ndarray of floats
        Vector containing the temporal entropy Ht of the selected files 
    M_array: 1d ndarray of floats
        Vector containing the amplitude index M  of the selected files 

    Returns
    -------    
    AR : 1d ndarray of floats
        Vector of acoustic richenss index
        
    References
    ----------
    .. [1] Depraetere, M., Pavoine, S., Jiguet, F., Gasc, A., Duvail, S., & Sueur, J. (2012). Monitoring animal diversity using acoustic indices: Implementation in a temperate woodland. Ecological Indicators, 13, 46–54.
    .. [2] Sueur, J., Aubin, T., & Simonis, C. (2008). Seewave: A free modular tool for sound analysis and synthesis. Bioacoustics, 18, 213–226.

    Examples:
    ---------
    >>> s, fs = maad.sound.load('../data/indices/S4A03895_20190522_060000.wav')
    >>> Ht_6h00 = maad.features.temporal_entropy(s)
    >>> M_6h00 = maad.features.temporal_median(s)
    
    >>> s, fs = maad.sound.load('../data/indices/S4A03895_20190522_080000.wav')
    >>> Ht_8h00= maad.features.temporal_entropy(s)
    >>> M_8h00 = maad.features.temporal_median(s)
    
    >>> s, fs = maad.sound.load('../data/indices/S4A03895_20190522_100000.wav')
    >>> Ht_10h00 = maad.features.temporal_entropy(s)
    >>> M_10h00 = maad.features.temporal_median(s)
    
    >>> maad.features.acoustic_richness_index([Ht_6h00,Ht_8h00,Ht_10h00],
                                              [M_6h00,M_8h00,M_10h00])
    array([0.11111111, 0.44444444, 1.        ])
    
    """    
    if len(Ht_array) != len(M_array) : 
        print ("warning : Ht_array and M_array must have the same length")
    
    AR = rankdata(Ht_array) * rankdata(M_array) / len(Ht_array)**2
    
    return AR

#=============================================================================

def temporal_activity (s, dB_threshold=3, mode='fast', Nt=512):
    """
    Compute the acoustic activity index in temporal domain.
    
    Acoustic activity corresponds to the portion of the waveform above a 
    threshold [1]_
    Three values are computed with this function:
        - ACTfract : proportion (fraction) of points above the threshold 
        - ACTcount : number of points above the threshold
        - ACTmean : mean value (in dB) of the portion of the signal above the threhold
    
    Parameters
    ----------
    s : 1D array of floats
        audio to process (wav)
    dB_threshold : scalar, optional, default is 3dB
        data >Threshold is considered to be an activity 
    mode : str, optional, default is "fast"
        Select the mode to compute the envelope of the audio waveform
        - "fast" : The sound is first divided into frames (2d) using the 
            function _wave2timeframes(s), then the max of each frame gives a 
            good approximation of the envelope.
        - "Hilbert" : estimation of the envelope from the Hilbert transform. 
            The method is slow
    Nt : integer, optional, default is 512
        Size of each frame. The largest, the highest is the approximation.    
    
    Returns
    -------    
    ACTfract :ndarray of scalars
        proportion (fraction) of points above the threshold for each frequency bin
    ACTcount: ndarray of scalars
        number of points above the threshold for each frequency bin
    ACTmean: scalar
        mean value (in dB) of the portion of the signal above the threhold
        
    References 
    ----------
    .. [1] Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived 
    from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> ACTfract, ACTcount, ACTmean = maad.features.temporal_activity (s, 6)
    >>> print('ACTfract: %2.2f / ACTcount: %2.0f / ACTmean: %2.2f' % (ACTfract, ACTcount, ACTmean))
    ACTfract: 0.37 / ACTcount: 620 / ACTmean: 24.41
    
    """ 

    ### For wave to be a ndarray
    s = np.asarray(s) 
    
    ### envelope
    if mode == 'fast' :
        env = envelope(s, mode='fast', Nt=Nt)
    elif mode == 'hilbert' :
        env = envelope(s, mode='hilbert')

    ### get background value
    _,BGNt,_ = temporal_snr(s, mode, Nt)
    
    # linear to power dB
    envdB = amplitude2dB(env)
    
    # subtract the background noise
    envdB = envdB - BGNt
    
    ACTtFraction, ACTtCount, ACTtMean = _acoustic_activity (envdB, dB_threshold, axis=0)
    
    return ACTtFraction, ACTtCount, ACTtMean

#=============================================================================
def temporal_events (s, fs, dB_threshold=3, rejectDuration=None, 
                  mode='fast', Nt=512, display=False, **kwargs):
    """
    Compute the acoustic event index from an audio signal.
    
    An acoustic event corresponds to the period of the signal above a 
    threshold. An acoustic event could be short (at list one point if 
    rejectDuration is None) or very long (the duration of the entire audio). 
    Two acoustic events are separated by a period with low audio signal (ie
    below the threshold)
    Four values are computed with this function:
        - EVNtFraction : Fraction: events duration over total duration
        - EVNmean : mean events duration (s)
        - EVNcount : number of events per s
        - EVN : binary vector or matrix with 1 corresponding to event position
    
    Parameters
    ----------
    s : 1D array of floats
        audio to process (wav)
    fs : Integer
        sampling frequency in Hz
    dB_threshold : scalar, optional, default is 3dB
        data >Threshold is considered to be an event 
        if the length is > rejectLength
    rejectDuration : scalar, optional, default is None
        event shorter than rejectDuration are discarded
        duration is in s
    mode : str, optional, default is "fast"
        Select the mode to compute the envelope of the audio waveform
        - "fast" : The sound is first divided into frames (2d) using the 
            function _wave2timeframes(s), then the max of each frame gives a 
            good approximation of the envelope.
        - "Hilbert" : estimation of the envelope from the Hilbert transform. 
            The method is slow
    Nt : integer, optional, default is 512
        Size of each frame. The largest, the highest is the approximation.
    display : boolean, optional, default is False
        Display the selected events on the audio waveform
    \*\*kwargs, optional. 
        This parameter is used by plt.plot

    Returns
    -------    
    EVNtFraction :scalar
        Fraction: events duration over total duration
    EVNmean: scalar
        mean events duration in s
    EVNcount: scalar
        number of events per s
    EVN: ndarray of floats 
        binary vector or matrix.
        1 corresponds to event
        0 corresponds to background

    References 
    ----------
    .. [1] Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived 
    from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> EVNtFract, EVNmean, EVNcount, _ = maad.features.temporal_events (s, fs, 6)
    >>> print('EVNtFract: %2.2f / EVNmean: %2.2f / EVNcount: %2.0f' % (EVNtFract, EVNmean, EVNcount))
    EVNtFract: 0.37 / EVNmean: 0.08 / EVNcount:  5
    
    """  
    ### For wave to be a ndarray
    s = np.asarray(s) 
    
    ### envelope
    if mode == 'fast' :
        env = envelope(s, mode, Nt)
        dt =1/fs*Nt
    elif mode == 'hilbert' :
        env = envelope(s, mode)
        dt = 1/fs
    
    # Time vector
    tn = np.arange(0,len(env),1)*len(s)/fs/len(env)
    
    ### get background value
    _,BGNt,_ = temporal_snr(s, mode, Nt)
    
    # linear to power dB
    envdB = 10*np.log10(env**2)
    
    # subtract the background noise
    envdB = envdB - BGNt
    
    EVNtSum, EVNtMean, EVNtCount, EVNt = _acoustic_events (envdB, dt, dB_threshold, rejectDuration=rejectDuration)
    
    # EVNtFraction
    EVNtFraction = EVNtSum / (dt*len(tn))
    
    ### display
    if display :
        fig, ax = plt.subplots()
        ax.plot(tn, env/max(abs(env)), lw=0.5, alpha=1)
        plt.fill_between(tn, 0, EVNt*1,color='red',alpha=0.5)
        ax.set_title('Detected Events')
        ax.set_xlabel('Time [sec]')   
    
    return EVNtFraction, EVNtMean, EVNtCount, EVNt


#******************************************************************************
#               FREQUENCY ECOACOUSTICS INDICES
#******************************************************************************

def frequency_entropy (X, compatibility="QUT") :
    """
    Computes the spectral entropy of a power spectral density (1d) or power
    spectrogram density (2d).

    Parameters
    ----------
    X : 1D or 2D array
        Power Spectral/Spectrogram Density (PSD) of an audio
        Better to work with PSD (amplitude¹) than with amplitude for energy 
        conservation
    compatibility : string {'QUT', 'seewave'}, default is 'QUT'
        Select the way to compute the spectral entropy.
            - QUT [2]_ : entropy of P
            - seewave [1]_ : entropy of sqrt(P)   
    Returns
    -------
    Hf: float
       spectral entropy of the audio 
    Ht_per_bin : array of floats
        temporal entropy along time axis for each frequency when P is a 
        spectrogram (2d) otherwise Ht_per_bin is empty   
       
    References
    ----------
    .. [1] Seewave : http://rug.mnhn.fr/seewave/
    Sueur, J., Aubin, T., & Simonis, C. (2008). Seewave, a free modular tool 
    for sound analysis and synthesis. Bioacoustics, 18(2), 213-226.
     
    .. [2] QUT : https://github.com/QutEcoacoustics/audio-analysis/
    Michael Towsey, Anthony Truskinger, Mark Cottman-Fields, & Paul Roe. 
    (2018, March 5). Ecoacoustics Audio Analysis Software v18.03.0.41 (Version v18.03.0.41). 
    Zenodo. http://doi.org/10.5281/zenodo.1188744
    
    Notes
    -----
    The spectral entropy of a signal measures the energy dispersion along frequencies. Low values 
    indicates a concentration of energy around a narrow frequency band. 
    If the DC value is not removed before processing the large peak at f=0Hz will 
    lower the entropy of the signal.

    Examples
    --------
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> Sxx_power,_,_,_ = maad.sound.spectrogram (s, fs)   
    >>> Hf, Ht_per_bin = maad.features.frequency_entropy(Sxx_power)
    >>> print(Hf)
    0.6313982665877063
    >>> print('Length of Ht_per_bin is : %2.0f' % len(Ht_per_bin))
    Length of Ht_per_bin is : 512
    >>> print(Ht_per_bin)
    [0.73458664 0.73476487 0.87981728 0.9161413  0.90153962 0.91684881
     0.91816039 0.93453925 0.92958317 0.93763948 0.93524745 0.93736222...]
    
    """
    # Force to be an array
    X = np.asarray(X)
    
    # test if P has 2 dimension (i.e a spectrogram Pxx)
    if X.ndim==1 :
        # Entropy
        if compatibility == 'QUT':
            Hf = entropy(X)
            Ht_per_bin = []
        elif compatibility == 'seewave':
            Hf = entropy(sqrt(X))
            Ht_per_bin = []
        else:
            raise TypeError('compatibility must be selected in {QUT, seewave}') 
    elif X.ndim==2 :     
        # Entropy
        if compatibility == 'QUT':
            Hf = entropy(mean(X, axis=1))
            # P is a spectrogram, computes entropy along time axis for each frequency
            Ht_per_bin = entropy(X, axis=1) 
        elif compatibility == 'seewave':
            Hf = entropy(sqrt(mean(X, axis=1)))
            # P is a spectrogram, computes entropy along time axis for each frequency
            Ht_per_bin = entropy(sqrt(X), axis=1)
        else:
            raise TypeError('compatibility must be selected in {QUT, seewave}')             
    else:
        raise TypeError('P must be a spectrum (1d) or a spectrogram (2d)')    
    
    return Hf, Ht_per_bin

#=============================================================================
def number_of_peaks(X, fn, mode='dB', min_peak_val=None, min_freq_dist=200, 
                  slopes=(1,1), display=False, **kwargs):
    """
    Count the number of frequency peaks on a mean spectrum.
    
    Parameters
    ----------
    X : ndarray of floats (1d) or (2d)
        Amplitude spectrum (1d) or spectrogram (2d). If spectrogram, the mean
        spectrum will be computed before finding peaks
    fn : 1d ndarray of floats
        frequency vector
    mode : string {dB, linear}, optional, default is dB
        select if the amplitude spectrum is converted into dB 
    min_peak_val : scalar, optional, default is None
        amplitude threshold parameter. Only peaks above this threshold will be 
        considered.
    min_freq_dist: scalar, optional, default is 200 
        frequency threshold parameter (in Hz). 
        If the frequency difference of two successive peaks is less than this threshold, 
        then the peak of highest amplitude will be kept only.
    slopes : tupple of two values, optional, default is (1,1)   
        slope parameter, tupple of float of length 2 corresponding to left and 
        right slopes, one or both could be set to None.
        Refers to the amplitude slopes of the peak. 
        The first value is the left slope and the second value is the right slope. 
        Only peaks with higher slopes than threshold values will be kept. 
    display: boolean, optional, default is False
        if True, display the mean spectrum with the detected peaks
        
    Returns
    -------
    NBPeaks : integer
        Number of detected peaks on the mean spectrum
    
    References
    ----------
    .. [1] Gasc, A. & al (2013). Biodiversity sampling using a global acoustic 
    approach: contrasting sites with microendemics in New Caledonia. 
    PloS one, 8(5), e65311.
    
    Inspired by the function `fpeaks` from the R package Seewave.
    .. [2] Sueur, J., Aubin, T., & Simonis, C. (2008). Seewave: A free modular tool for sound 
    analysis and synthesis. Bioacoustics, 18, 213–226.
    
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power, tn, fn, _ = maad.sound.spectrogram (s, fs)  
    >>> maad.features.number_of_peaks(Sxx_power, fn, slopes=6, min_freq_dist=100, display=True) 

    """
    # Force to be an array
    X = np.asarray(X)
    
    # mean spectrum
    if X.ndim == 2 :
        S = avg_amplitude_spectro(X)
    else:
        S = X
    
    # if mode is "dB", convert into dB
    if mode == 'dB' :
        S = amplitude2dB(S)
        if min_peak_val is not None :
            min_peak_val = amplitude2dB(min_peak_val)

    # Find peaks
    min_pix_distance = min_freq_dist/(fn[1]-fn[0])
    index, prop = find_peaks(S, height = min_peak_val, 
                             distance = min_pix_distance, 
                             prominence=0)
    
    # keep peaks with with slopes higher than the limit
    if slopes is None :
        index_select = index
    elif isinstance(slopes, numbers.Number) :
        left_slope = S[index] - S[prop['left_bases']]
        index_select = index[(left_slope>=slopes)]
    elif len(slopes) == 2 :
        if (slopes[0] is not None) and (slopes[1] is not None)   :
            left_slope = S[index] - S[prop['left_bases']]
            right_slope = S[index] - S[prop['right_bases']]
            index_select = index[(left_slope>=slopes[0]) & (right_slope>slopes[1])]
        elif (slopes[0] is not None) and (slopes[1] is None) :
            left_slope = S[index] - S[prop['left_bases']]
            index_select = index[(left_slope>=slopes[0])]
        elif (slopes[0] is None) and (slopes[1] is not None) :
            right_slope = S[index] - S[prop['right_bases']]
            index_select = index[(right_slope>slopes[1])]
        else:
            index_select = index
    else:
        index_select = index
    
    # number of peaks
    NBPeaks = len(index_select)
    
    # display
    if display :
        if mode == 'dB' :
            ylabel ='Amplitude [dB]'
        else:
            ylabel = 'Amplitude [AU]'
        fig_kwargs = {
                      'figtitle':'Mean Spectrum with detected peaks',
                      'xlabel': kwargs.pop('xlabel','Frequency [Hz]'),
                      'ylabel': kwargs.pop('ylabel',ylabel)
                      }

        ax, _ = plot1d(fn,S, **fig_kwargs)
        ax.plot(fn[index_select], S[index_select], '+', mfc=None, mec='r', 
                mew=2, ms=8)
    return NBPeaks


#=============================================================================

####    Indices based on the entropy

def spectral_entropy (Sxx, fn, flim=None, display=False) :
    """
    Compute different entropies based on the average spectrum, its variance, 
    and its maxima [1]_     
    
    Parameters
    ----------
    Sxx : ndarray of floats
        Spectrogram (2d). 
        It is recommended to work with PSD to be consistent with energy conservation
    
    fn : 1d ndarray of floats
        frequency vector
    
    flim : tupple (fmin, fmax), optional, default is None
        Frequency band used to compute the spectral entropy.
        For instance, one may want to compute the spectral entropy for the 
        biophony bandwidth
    
    display : boolean, optional, default is False
        Display the different spectra (mean, variance, covariance, max...)
        
    Returns
    -------     
    EAS : scalar
        Entropy of Average Spectrum
    ECU : scalar
        Entropy of spectral variance (along the time axis for each frequency)
    ECV : scalar
        Entropy of Coefficient of Variation (along the time axis for each frequency)
    EPS : scalar
        Entropy of spectral maxima (peaks) 
    EPS_KURT : scalar
        Kurtosis of spectral maxima
    EPS_SKEW : scalar
        Skewness of spectral maxima
        
    References 
    ----------
    Credit : Towsey 2017
    
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power, tn, fn, _ = maad.sound.spectrogram (s, fs)  
    >>> EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW = maad.features.spectral_entropy(Sxx_power, fn, flim=(2000,10000)) 
    >>> print('EAS: %2.2f / ECU: %2.2f / ECV: %2.2f / EPS: %2.2f / EPS_KURT: %2.2f / EPS_SKEW: %2.2f' % (EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW))
    EAS: 0.27 / ECU: 0.49 / ECV: 0.24 / EPS: 1.00 / EPS_KURT: 17.58 / EPS_SKEW: 3.55
    
    """
    
    if isinstance(flim, numbers.Number) :
        print ("WARNING: flim must be a tupple (fmin, fmax) or None")
        return
    
    if flim is None : flim=(fn.min(),fn.max())
    
    # select the indices corresponding to the frequency range
    iBAND = index_bw(fn, flim)
    
    # force Sxx to be an ndarray
    X = np.asarray(Sxx)

    # TOWSEY & BUXTON : only on the bio band
    # EAS [TOWSEY] #
    ####  COMMENT : Result a bit different due to different Hilbert implementation
    X_mean = mean(X[iBAND], axis=1)
    Hf = entropy(X_mean)
    EAS = 1 - Hf

    #### Entropy of spectral variance (along the time axis for each frequency)
    """ ECU [TOWSEY] """
    X_Var = var(X[iBAND], axis=1)
    Hf_var = entropy(X_Var)
    ECU = 1 - Hf_var

    #### Entropy of coefficient of variance (along the time axis for each frequency)
    """ ECV [TOWSEY] """
    X_CoV = var(X[iBAND], axis=1)/mean(X[iBAND], axis=1)
    Hf_CoV = entropy(X_CoV)
    ECV = 1 - Hf_CoV
    
    #### Entropy of spectral maxima 
    """ EPS [TOWSEY]  """
    ioffset = np.argmax(iBAND==True)
    Nbins = sum(iBAND==True)  
    imax_X = np.argmax(X[iBAND],axis=0) + ioffset
    imax_X = fn[imax_X]
    max_X_bin, bin_edges = np.histogram(imax_X, bins=Nbins, range=flim)
    
    if sum(max_X_bin) == 0 :
        max_X_bin = np.zeros(len(max_X_bin))
        EPS = float('nan')
        #### Kurtosis of spectral maxima
        EPS_KURT = float('nan')
        #### skewness of spectral maxima
        EPS_SKEW = float('nan')
    else:
        max_X_bin = max_X_bin/sum(max_X_bin)
        Hf_fmax = entropy(max_X_bin)
        EPS = 1 - Hf_fmax    
        #### Kurtosis of spectral maxima
        EPS_KURT = kurtosis(max_X_bin)
        #### skewness of spectral maxima
        EPS_SKEW = skewness(max_X_bin)
    
    if display: 
        fig, ax = plt.subplots()
        ax.plot(fn[iBAND], X_mean/max(X_mean),label="Normalized mean")
        plt.plot(fn[iBAND], X_Var/max(X_Var),label="Normalized variance")
        ax.plot(fn[iBAND], X_CoV/max(X_CoV),label="Normalized covariance")
        ax.plot(fn[iBAND], max_X_bin/max(max_X_bin),label="Normalized Spectral max")
        ax.set_title('Signals')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend()

    return EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW

#=============================================================================

def spectral_cover (Sxx, fn, dB_threshold=3, flim_LF=(0,1000), flim_MF=(1000,10000), 
                   flim_HF=(10000,20000)):
    """
    Compute the proportion (cover) of the spectrogram above a threshold for 
    three bandwidths : low frequency band (LF), medium frequency band (MF) and 
    high frequency band (HF).  
    
    Parameters
    ----------
    Sxx : 2D array of floats
        Spectrogram 2D in dB. Usually, better to work with spectrogram without 
        stationnary noise in order to measure only acoustic activity above the
        background noise
    fn : 1d ndarray of floats
        frequency vector 
    dB_threshold : scalar, optional, default is 3dB
        data >Threshold is considered to be an activity 
    flim_LF : tupple, optional, default is (0,1000)
        Low frequency band in Hz
    flim_MF : tupple, optional, default is (1000,10000)
        mid frequency band in Hz    
    flim_HF : tupple, optional, default is (10000,20000)
        high frequency band in Hz
    
    Returns
    -------    
    LFC :scalar
        Proportion of the LF bandwidth of the spectrogram with activity above 
        the threshold
    MFC: scalar
        Proportion of the MF bandwidth of the spectrogram with activity above 
        the threshold
    HFC: scalar
        Proportion of the HF bandwidth of the spectrogram with activity above 
        the threshold
        
    References 
    ----------
    .. [1] Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived 
    from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    
    Examples :
    ----------
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power, tn, fn, ext = maad.sound.spectrogram (s, fs)  
    >>> Sxx_noNoise= maad.sound.median_equalizer(Sxx_power, display=True, extent=ext) 
    >>> Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)
    >>> LFC, MFC, HFC = maad.features.spectral_cover(Sxx_dB_noNoise, fn) 
    >>> print('LFC: %2.2f / MFC: %2.2f / HFC: %2.2f' % (LFC, MFC, HFC))
    LFC: 0.15 / MFC: 0.19 / HFC: 0.13
    
    """ 

    ### For Sxx to be a ndarray
    Sxx = np.asarray(Sxx) 
    
    idx = index_bw(fn,flim_LF)
    lowFreqCover, _, _ = _acoustic_activity (Sxx[idx], dB_threshold, axis=1)
    LFC = mean(lowFreqCover)
    
    idx = index_bw(fn,flim_MF)
    midFreqCover, _, _ = _acoustic_activity (Sxx[idx], dB_threshold, axis=1)
    MFC = mean(midFreqCover)
    
    idx = index_bw(fn,flim_HF)
    highFreqCover, _, _ = _acoustic_activity (Sxx[idx], dB_threshold, axis=1)
    HFC = mean(highFreqCover)
    
    return LFC, MFC, HFC


#=============================================================================

def spectral_activity (Sxx_dB, dB_threshold=6):
    """
    Compute the acoustic activity on a spectrogram.
    
    Acoustic activity corresponds to the portion of the spectrogram above a 
    threshold frequency per frequency along time axis [1]_
    The function computes for each frequency bin:
        - ACTfract : proportion (fraction) of points above the threshold 
        - ACTcount : number of points above the threshold
        - ACTmean : mean value (in dB) of the portion of the signal above the threhold
    
    Parameters
    ----------
    Sxx_dB : 2D array of floats
        Spectrogram 2D in dB. Usually, better to work with spectrogram without 
        stationnary noise in order to measure only acoustic activity above the
        background noise
    dB_threshold : scalar, optional, default is 6dB
        data >Threshold is considered to be an activity 
    
    Returns
    -------    
    ACTspfract :ndarray of scalars
        proportion (fraction) of points above the threshold for each frequency bin
    ACTspcount: ndarray of scalars
        number of points above the threshold for each frequency bin
    ACTspmean: scalar
        mean value (in dB) of the portion of the signal above the threhold
        
    References 
    ----------
    Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived 
    from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    
    Examples
    --------
    >>> import numpy as np
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power, tn, fn, ext = maad.sound.spectrogram (s, fs)  
    >>> Sxx_noNoise= maad.sound.median_equalizer(Sxx_power, display=True, extent=ext) 
    >>> Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)
    >>> ACTspfract_per_bin, ACTspcount_per_bin, ACTspmean_per_bin = maad.features.spectral_activity(Sxx_dB_noNoise)  
    >>> print('Mean proportion of spectrogram above threshold : %2.2f%%' %np.mean(ACTspfract_per_bin))
    Mean proportion of spectrogram above threshold : 0.07%
    
    """ 

    ### For Sxx_dB to be a ndarray
    Sxx_dB = np.asarray(Sxx_dB) 
    
    ACTspfract, ACTspcount, ACTspmean = _acoustic_activity (Sxx_dB, dB_threshold, 
                                                            axis=1)
    
    return ACTspfract, ACTspcount, ACTspmean

#=============================================================================
def spectral_events (Sxx_dB, dt, dB_threshold=6, rejectDuration=None, 
                     display=False, **kwargs):
    """
    Compute acoustic events from a spectrogram [1]_.
    
    An acoustic event corresponds to the period of the signal above a 
    threshold. An acoustic event could be short (at list one point if 
    rejectDuration is None) or very long (the duration of the entire audio). 
    Two acoustic events are separated by a period with low audio signal (ie
    below the threshold). Acoustic events are calculated frequency by frequency
    along time axis
    This function computes:
        - EVNspFraction : Fraction of events duration over total duration
        - EVNspmean : mean events duration (s)
        - EVNspcount : number of events per s
        - EVNsp : binary vector or matrix with 1 corresponding to event position
    
    Parameters
    ----------
    Sxx_dB : 2D array of floats
        2D in dB. Usually, better to work with spectrogram without 
        stationnary noise in order to measure only acoustic activity above the
        background noise
    dt : float
        time resolution in s (ie tn[1]-tn[0])
    dB_threshold : scalar, optional, default is 6dB
        data >Threshold is considered to be an event 
        if the length is > rejectLength
    rejectDuration : scalar, optional, default is None
        event shorter than rejectDuration are discarded
        duration is in s
    display : boolean, optional, default is false
        Display a plot with the number of events per s (EVNspCount) and
        a binary image with the detected events.
    \*\*kwargs : optional. See matplotlib documentation

    Returns
    -------    
    EVNspFract :scalar
        Fraction: events duration over total duration
    EVNspMean: scalar
        mean events duration in s
    EVNspCount: scalar
        number of events per s
    EVNsp: ndarray of floats 
        binary matrix.
        1 corresponds to event
        0 corresponds to background

    References 
    ----------
    .. [1] Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived 
    from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    
    Examples
    --------
    >>> import numpy as np
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power, tn, fn, ext = maad.sound.spectrogram (s, fs)  
    >>> Sxx_noNoise= maad.sound.median_equalizer(Sxx_power) 
    >>> Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)
    >>> EVNspFract_per_bin, EVNspMean_per_bin, EVNspCount_per_bin, EVNsp = maad.features.spectral_events(Sxx_dB_noNoise, dt=tn[1]-tn[0], dB_threshold=6, rejectDuration=0.1, display=True, extent=ext)  
    >>> print('Mean proportion of spectrogram with event s: %2.2f%%' %np.mean(EVNspFract_per_bin))
    Mean proportion of spectrogram with events : 0.01%
    
    """  
    ### For wave to be a ndarray
    Sxx_dB = np.asarray(Sxx_dB) 
        
    EVNspSum, EVNspMean, EVNspCount, EVNsp = _acoustic_events (Sxx_dB, dt, 
                                                               dB_threshold, 
                                                               rejectDuration=rejectDuration)
    
    # EVNspFract = EVNspSum  * total_duration
    EVNspFract = EVNspSum / (dt * Sxx_dB.shape[1])
    
    if display :
        # display Number of events/s / frequency
        extent =  kwargs.pop('extent',None)
        if extent is not None : 
            y = np.arange(0, Sxx_dB.shape[0])/Sxx_dB.shape[0]*extent[3]
            xlabel = 'frequency [Hz]' 
        else: 
            y = np.arange(0, Sxx_dB.shape[0])
            xlabel = 'pseudofrequency [points]'   

        fig1, ax1 = plt.subplots()
        plt.plot(y, EVNspCount)
        ax1.set_xlabel(xlabel)
        ax1.set_title('EVNspCount : Number of events/s')
        
    # display EVENTS detected in the spectrogram
        if extent is not None :
            xlabel = 'Time [sec]'
            ylabel = 'frequency [Hz]' 
        else: 
            extent = (0,Sxx_dB.shape[1],0,Sxx_dB.shape[0])
            xlabel = 'pseudoTime [sec]'
            ylabel = 'pseudofrequency [points]'   
    
        # set the paramters of the figure
        title  =kwargs.pop('title','Events detected') 
        cmap   =kwargs.pop('cmap','gray')  
        figsize=kwargs.pop('figsize',(4, 13))  
        vmin=kwargs.pop('vmin',0)  
        vmax=kwargs.pop('vmax',1)
        
        ax, fig = plot2d (EVNsp*1, extent=extent, now=False, figsize=figsize, 
                          title=title, ylabel=ylabel,xlabel=xlabel,
                          vmin=vmin,vmax=vmax,  cmap=cmap, **kwargs) 
    
    return EVNspFract, EVNspMean, EVNspCount, EVNsp


#=============================================================================
def acoustic_complexity_index(Sxx):
    
    """
    Compute the Acoustic Complexity Index (ACI) from a spectrogram [1]_.
        
    Parameters
    ----------
    Sxx : ndarray of floats
        2d : Spectrogram (i.e matrix of spectrum)
    
    Returns
    -------    
    ACI_xx: 2d ndarray of scalars
        Acoustic Complexity Index of the spectrogram
    
    ACI_per_bin: 1d ndarray of scalars
        ACI value for each frequency bin
        sum(ACI_xx,axis=1)
        
    ACI_sum: scalar
        Sum of ACI value per frequency bin (Common definition)
        sum(ACI_per_bin)
    
    Notes
    -----   
    ACI depends on the duration of the spectrogram as the derivation of the signal
    is normalized by the sum of the signal. 
    Thus, if the background noise is high due to high acoustic activity the
    normalization by the sum of the signal reduced ACI.
    So ACI is low when there is no acoustic activity or high acoustic activity 
    with continuous background noise.
    ACI is high only when acoustic activity is medium, with sounds well above
    the background noise.
        
    References
    ----------
    .. [1] Pieretti N, Farina A, Morri FD (2011) A new methodology to infer the singing 
    activity of an avian community: the Acoustic Complexity Index (ACI). 
    Ecological Indicators, 11, 868-873.
    
    
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx, tn, fn, ext = maad.sound.spectrogram (s, fs, mode='amplitude')  
    >>> _, _ , ACI  = maad.features.acoustic_complexity_index(Sxx)
    >>> print('ACI : %2.0f ' %ACI)
    ACI : 306

    """   
    ACI_xx = ((np.abs(diff(Sxx,1)).transpose())/(np.sum(Sxx,1)).transpose()).transpose()       
    ACI_per_bin = np.sum(ACI_xx,axis=1)
    ACI_sum = np.sum(ACI_per_bin)
    
    return ACI_xx, ACI_per_bin, ACI_sum 

#=============================================================================
def acoustic_diversity_index (Sxx, fn, fmin=0, fmax=20000, bin_step=1000, 
                            dB_threshold=-50, index="shannon"):
    
    """
    Compute the Acoustic Diversity Index (ADI) from a spectrogram [1]_.
    
    The diversity can be computed using Shannon, Simpson, or the inverse Simpson diversity index.
    
    Parameters
    ----------
    Sxx : ndarray of floats
        2d : Spectrogram
    
    fn : 1d ndarray of floats
        frequency vector
    
    fmin : scalar, optional, default is 0
        Minimum frequency in Hz
        
    fmax : scalar, optional, default is 20000
        Maximum frequency in Hz
        
    bin_step : scalar, optional, default is 500
        Frequency step in Hz
    
    dB_threshold : scalar, optional, default is -50dB
        Threshold to compute the score (ie. the number of data > threshold,
        normalized by the length)
        
    index : string, optional, default is "shannon"
        - "shannon" : Shannon entropy is calculated on the vector of scores
        - "simpson" : Simpson index is calculated on the vector of scores
        - "invsimpson" : Inverse Simpson index is calculated on the vector of scores
    
    Returns
    -------    
    ADI : scalar 
        Acoustic Diversity Index of the spectrogram (ie. index of the vector 
        of scores)
    
    Notes
    -----
    The Acoustic Eveness Index (AEI) and the Acoustic Diversity Index (ADI) are negatively correlated.
    
    See also
    --------
    acoustic_eveness_index
    
    References
    ----------
    .. [1] Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. 
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx, tn, fn, ext = maad.sound.spectrogram (s, fs, mode='amplitude')  
    >>> ADI  = maad.features.acoustic_diversity_index(Sxx,fn)
    >>> print('ADI : %2.2f ' %ADI)
    ADI : 2.45 
    
    """
        
    # number of frequency intervals to compute the score
    N = np.floor((fmax-fmin)/bin_step)
    
    # convert into dB and normalization by the max
    Sxx_dB = amplitude2dB(Sxx/max(Sxx))       
    
    # Score for each frequency in the frequency bandwith
    s_sum = []
    for ii in np.arange(0,N):
        f0 = int(fmin+bin_step*(ii))
        f1 = int(f0+bin_step)
        s,_ = _score(Sxx_dB[index_bw(fn,(f0,f1)),:], threshold=dB_threshold, axis=0)
        s_sum.append(mean(s))
    
    s = np.asarray(s_sum)
    
    # Entropy
    if index =="shannon":
        ADI = _shannonEntropy(s)
    elif index == "simpson":
        s = s/sum(s)
        s = s**2
        ADI = 1-sum(s)
    elif index == "invsimpson":
        s = s/sum(s)
        s = s**2
        ADI = 1/sum(s)   
    
    return ADI

#=============================================================================
def acoustic_eveness_index (Sxx, fn, fmin=0, fmax=20000, bin_step=500, 
                          dB_threshold=-50):
    
    """
    Compute the Acoustic Eveness Index (AEI) from a spectrogram [1]_.
    
    Parameters
    ----------
    Sxx: ndarray of floats
        2d : Spectrogram
    
    fn : 1d ndarray of floats
        frequency vector
    
    fmin : scalar, optional, default is 0
        Minimum frequency in Hz
        
    fmax : scalar, optional, default is 20000
        Maximum frequency in Hz
        
    bin_step : scalar, optional, default is 500
        Frequency step in Hz
    
    dB_threshold : scalar, optional, default is -50
        Threshold to compute the score (ie. the number of data > threshold,
        normalized by the length)
        
    Returns
    -------    
    AEI : scalar 
        Acoustic Eveness of the spectrogram (ie. Gini of the vector of scores)
        
    Notes
    -----
    The Acoustic Eveness Index (AEI) and the Acoustic Diversity Index (ADI) are negatively correlated.
    
    See also
    --------
    acoustic_diversity_index
        
    References 
    ----------
    .. [1] Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. 
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx, tn, fn, ext = maad.sound.spectrogram (s, fs, mode='amplitude')  
    >>> AEI  = maad.features.acoustic_eveness_index(Sxx,fn)
    >>> print('AEI : %2.2f ' %AEI)
    AEI : 0.56    
    
    """

    # number of frequency intervals to compute the score
    N = np.floor((fmax-fmin)/bin_step)
    
    # convert into dB and normalization by the max
    Sxx_dB = amplitude2dB(Sxx/max(Sxx))
 
    # Score for each frequency in the frequency bandwith
    s_sum = []
    for ii in np.arange(0,N):
        f0 = int(fmin+bin_step*(ii))
        f1 = int(f0+bin_step)
        s,_ = _score(Sxx_dB[index_bw(fn,(f0,f1)),:], threshold=dB_threshold, axis=0)
        s_sum.append(mean(s))
    
    s = np.asarray(s_sum)
    
    # Gini
    AEI = _gini(s)
    
    return AEI

#=============================================================================
####    Indices based on the energy
#=============================================================================
def soundscape_index (Sxx_power,fn,flim_bioPh=(1000,10000),flim_antroPh=(0,1000), 
                     R_compatible = 'soundecology'):
    """
    Compute the Normalized Difference Soundscape Index from a power spectrogram [1]_.
        
    Parameters
    ----------
    Sxx_power : ndarray of floats
        2d : Power Spectrogram
    
    fn : vector
        frequency vector 
        
    flim_bioPh : tupple (fmin, fmax), optional, default is (1000,10000)
        Frequency band of the biophony
    
    flim_antroPh: tupple (fmin, fmax), optional, default is (0,1000)
        Frequency band of the anthropophony
    
    R_compatible : string, optional, default is "soundecology"
        if 'soundecology', the result is similar to the package SoundEcology in R 
        Otherwise, the result is specific to maad or Seewave R package
        
    Returns
    -------
    NDSI : scalar
        (bioPh-antroPh)/(bioPh+antroPh)
    ratioBA : scalar
        biophonic energy / anthropophonic energy
    antroPh : scalar
        Acoustic energy in the anthropophonic bandwidth
    bioPh : scalar
        Acoustic energy in the biophonic bandwidth
    
    References
    ----------
    .. [1] Kasten, Eric P., Stuart H. Gage, Jordan Fox, and Wooyeong Joo. 2012. 
    The Remote Environmental Assessment Laboratory's Acoustic Library: An Archive 
    for Studying Soundscape Ecology. Ecological Informatics 12: 50-67.
    
    Inspired by Seewave and soundecology R packages.
    
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power, tn, fn, ext = maad.sound.spectrogram (s, fs)  
    >>> NDSI, ratioBA, antroPh, bioPh  = maad.features.soundscape_index(Sxx_power,fn)
    >>> print('NDSI Soundecology : %2.2f ' %NDSI)
    NDSI Soundecology : 0.10
    >>> NDSI, ratioBA, antroPh, bioPh  = maad.features.soundscape_index(Sxx_power,fn,R_compatible=None)
    >>> print('NDSI MAAD: %2.2f ' %NDSI)
    NDSI MAAD : 0.99
    
    """
    
    if R_compatible == 'soundecology' :
        # Step is determined as the difference between anthro_max and anthro_min
        bin_step = flim_antroPh[1] - flim_antroPh[0]
        #Convert into bins
        Sxx_bins, bins = into_bins(Sxx_power, fn, bin_min=fn[0], bin_max=fn[-1], 
                                  bin_step=bin_step, axis=0)   
    else:
        # Frequency resolution is 1000 Hz
        bin_step = 1000
        #Convert into bins
        Sxx_bins, bins = into_bins(Sxx_power, fn, bin_min=fn[0], bin_max=fn[-1], 
                                  bin_step=bin_step, axis=0) 
        # In Seewave, the first bin (0kHz) is removed
        Sxx_bins = Sxx_bins[bins>=1000,]
        bins = bins[bins>=1000]
        
    # Energy in BIOBAND
    bioPh = sum(Sxx_bins[index_bw(bins, flim_bioPh), ])
    # Energy in ANTHROPOBAND
    antroPh = sum(Sxx_bins[index_bw(bins, flim_antroPh), ])
    
    # NDSI and ratioBA indices 
    NDSI = (bioPh-antroPh)/(bioPh+antroPh)
    ratioBA = bioPh / antroPh

    return NDSI, ratioBA, antroPh, bioPh

#=============================================================================
def bioacoustics_index (Sxx, fn, flim=(2000, 15000), R_compatible ='soundecology'):
    """
    Compute the Bioacoustics Index from a spectrogram [1]_.
    
    Parameters
    ----------
    Sxx : ndarray of floats
        matrix : Spectrogram  
    fn : vector
        frequency vector 
    flim : tupple (fmin, fmax), optional, default is (2000, 15000)
        Frequency band used to compute the bioacoustic index.
    R_compatible : string, default is "soundecology"
        if 'soundecology', the result is similar to the package SoundEcology in R 
        Otherwise, the result is specific to maad
    
    Returns
    -------
    BI : scalar
        Bioacoustics Index
    
    References 
    ----------
    .. [1] Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic 
    invasion resistance in Hawaii: bioacoustics, field surveys, and airborne 
    remote sensing. Ecological Applications 17: 2137-2144.
    
    Ported and modified from the soundecology R package.
    
    Notes
    -----    
    Soundecology compatible version:
        - average of dB value
        - remove negative value in order to get positive values only
        - dividing by the frequency resolution df instead of multiplication
    
    Examples :
    ----------
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx, tn, fn, ext = maad.sound.spectrogram (s, fs,mode='amplitude')  
    >>> BI = maad.features.bioacoustics_index(Sxx,fn)
    >>> print('BI Soundecology : %2.2f ' %BI)
    BI Soundecology : 52.84
    >>> BI  = maad.features.bioacoustics_index(Sxx,fn,R_compatible=None)
    >>> print('BI MAAD: %2.2f ' %BI)
    BI MAAD : 17.05
    """    
    
    # select the indices corresponding to the frequency bins range
    indf = index_bw(fn,flim)
    
    # frequency resolution. 
    df = fn[1] - fn[0]
    
    # ======= As soundecology
    if R_compatible == 'soundecology' :
        # Mean Sxx normalized by the max
        meanSxx = mean(Sxx/max(Sxx), axis=1)
        # Convert into dB
        meanSxxdB = amplitude2dB(meanSxx)
        
        # "normalization" in order to get positive 'vectical' values 
        meanSxxdB = meanSxxdB[indf,]-min(meanSxxdB[indf,])
    
        # this is not the area under the curve...
        # what is the meaning of an area under the curve in dB...
        BI = sum(meanSxxdB)/df
    # ======= maad version    
    else:
        # better to average the PSD for energy conservation
        PSDxx_norm = (Sxx**2/max(Sxx**2))
        meanPSDxx_norm = mean(PSDxx_norm, axis=1)

        # Compute the area
        # take the sqrt in order to go back to Sxx
        BI = sqrt(sum(meanPSDxx_norm))* df 
        
    return BI
        
#=============================================================================
#       
#   New ecoacoustics indices introduced by S. HAUPERT, 2020
#   
#============================================================================= 

#=============================================================================

def temporal_leq (s, fs, gain, Vadc=2, sensitivity=-35, dBref=94, dt=1): 
    """
    Computes the Equivalent Continuous Sound level (Leq) of an audio signal 
    in the time domain.

    Parameters
    ----------
    s : 1D array of floats
        audio to process (wav)
    fs : Integer
        sampling frequency in Hz
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
    Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
        Maximal voltage (peak to peak) converted by the analog to digital convertor ADC    
    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)
    dt : float, optional, default is 1 (second)
        Integration step to compute the Leq (Equivalent Continuous Sound level)
    
    Returns
    -------
    LEQt: float
        Equivalent Continuous Sound level (Leq) in dB SPL

    Examples
    --------
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> Leq = maad.features.temporal_leq (s, fs, gain=42)
    >>> print('Leq is %2.1fdB SPL' % Leq)
    Leq is 63.7dB SPL
    
    """
    # compute the Leq for each dt step
    leq = wav2leq(s, fs, gain, Vadc, dt, sensitivity, dBref)
    # average them
    LEQt = mean_dB(leq, axis=1)
    
    return LEQt

#=============================================================================

def spectral_leq (X, gain, Vadc=2, sensitivity=-35, dBref=94, pRef = 20e-6): 
    """
    Computes the Equivalent Continuous Sound level (Leq) from a power spectrum 
    (1d) or power spectrogram (2d).

    Parameters
    ----------
    X : ndarray of floats
        Spectrum (1d) or Spectrogram (2d). 
        Work with PSD to be consistent with energy concervation
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
    Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
        Maximal voltage (peak to peak) converted by the analog to digital convertor ADC    
    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)
    pRef : Sound pressure reference in the medium (air : 20e-6, water : 1e-6)
    
    Returns
    -------
    LEQf: float
        Equivalent Continuous Sound level (Leq) in dB SPL

    Examples
    --------
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> Sxx_power,_,_,_ = maad.sound.spectrogram(s,fs)
    >>> Leqf, Leqf_per_bin = maad.features.spectral_leq(Sxx_power, gain=42)
    >>> print('Leq (from spectrogram) is %2.1fdB SPL' % Leqf)
    Leq (from spectrogram) is 63.7dB SPL
    
    """
    # force X to be an ndarray
    X = np.asarray(X)
    
    # test if X has 2d (Spectrogram Pxx)
    if X.ndim == 2 : 
        # average spectrogram along time direction
        X = mean(X, axis=1)
        # convert power spectrogram/spectrum into dBSPL
        LEQf_per_bin = power2dBSPL(X, gain, Vadc, sensitivity, dBref, pRef)
    else :
        LEQf_per_bin = []
        
    # convert spectrogram/spectrum into pressure
    LEQf = psd2leq(X, gain, Vadc, sensitivity, dBref, pRef)
    
    return LEQf, LEQf_per_bin

#=============================================================================

def more_entropy(x, order=3, axis=0) :
    """
    Compute the entropy of an audio signal using multiple methods.
    
    There are currently five types supported:
        - Havrda
        - Renyi
        - paired Shannon
        - gamma
        - Gini Simpson
        
    Parameters
    ----------
    x : ndarray of floats 
        vector (1d) or matrix (2d) of scalars.
        Vector could be audio recording or spectrum
        Matrix could be spectrogram
    order : integer, default is 3
        determine the order of the entropy in case of Havrda, Renyi and gamma 
        entropy. if order =2, Havrda is equal to Gini Simpson entropy
    axis : integer, default is 0
        In case of x is a matrix, select the row (axis=0) or the columns (axis=1)
        of the matrix to compute the entropies.  
    
    Returns
    -------
    H_Havrda : scalar
        Havrda entropy
    H_Renyi : scalar
        Renyi entropy
    H_pairedShannon : scalar
        Paired Shannon entropy
    H_gamma : scalar
        Gamma entropy
    H_GiniSimpson : scalar
        Gini Simpson entropy
        
    References
    ----------
    1. Zhao, Yueqin. "Rao's Quadratic Entropy and Some New Applications" (2010). 
    Doctor of Philosophy (PhD), dissertation,Mathematics and Statistics, 
    Old Dominion University, DOI: 10.25777/qgak-sf09
    
    Examples
    --------
    
    Compute entropy in time domain.
    
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> env = maad.sound.envelope(s)
    >>> Ht_Havrda, Ht_Renyi, Ht_pairedShannon, Ht_gamma, Ht_GiniSimpson = maad.features.more_entropy(env**2, order=3)
    >>> print('Ht_Havrda: %2.2f / Ht_Renyi: %2.2f / Ht_pairedShannon: %2.2f / Ht_gamma: %2.0f / Ht_GiniSimpson: %2.2f' % (Ht_Havrda, Ht_Renyi, Ht_pairedShannon, Ht_gamma, Ht_GiniSimpson))
    Ht_Havrda: 0.33 / Ht_Renyi: 7.20 / Ht_pairedShannon: 9.04 / Ht_gamma: 24223924 / Ht_GiniSimpson: 1.00
    
    Compute entropy in spectral domain.
    
    >>> Sxx_power,_,_,_ = maad.sound.spectrogram(s,fs)
    >>> S_power = maad.sound.avg_power_spectro(Sxx_power)
    >>> Hf_Havrda, Hf_Renyi, Hf_pairedShannon, Hf_gamma, Hf_GiniSimpson = maad.features.more_entropy(S_power, order=3)
    >>> print('Hf_Havrda: %2.2f / Hf_Renyi: %2.2f / Hf_pairedShannon: %2.2f / Hf_gamma: %2.0f / Hf_GiniSimpson: %2.2f' % (Hf_Havrda, Hf_Renyi, Hf_pairedShannon, Hf_gamma, Hf_GiniSimpson))
    Hf_Havrda: 0.33 / Hf_Renyi: 3.23 / Hf_pairedShannon: 4.92 / Hf_gamma: 7931 / Hf_GiniSimpson: 0.97
    
    """
    
    if isinstance(x, (np.ndarray)) == True:
        if x.ndim > axis:
            if x.shape[axis] == 1:
                axis=0
                print ("WARNING: axis is to large, axis is set to 0") 
            # if datain contains negative values -> rescale the signal between 
            # between posSitive values (for example (0,1))
            if np.min(x)<0:
                x = linear_scale(x,minval=0,maxval=1)
            # Tranform the signal into a Probability mass function (pmf)
            # Sum(pmf) = 1
            if axis == 0 :
                pmf = x/np.sum(x,axis)
            elif axis == 1 :                     
                pmf = (x.transpose()/np.sum(x,axis)).transpose()
            pmf[pmf==0] = _MIN_
            # alpha order entropy of Havrda and Charvat
            H_Havrda = (1-np.sum(pmf**order, axis=axis)) / (2**(order-1)-1)
            # alpha order entropy of Renyi
            H_Renyi = np.log(np.sum(pmf**order, axis=axis))/(1-order)
            # paired Shannon entropy
            H_pairedShannon = -np.sum(pmf*log(pmf), axis=axis)-np.sum((1-pmf)*log(1-pmf), axis=axis)
            # gamma entropy
            H_gamma = (1-(np.sum(pmf**(1/order), axis=axis))**order)/(1-2**(order-1))
            # Gini-Simpson entropy
            H_GiniSimpson = 1-np.sum(pmf**2,axis=axis)           
                
    return H_Havrda, H_Renyi, H_pairedShannon, H_gamma, H_GiniSimpson

#=============================================================================

def frequency_raoq (S_power, fn, bin_step=1000):
    """
    Compute Rao's quadratic entropy on a power spectrum (1d).
        
    Parameters
    ----------
    S_power : ndarray of floats 
        Spectrum (1d)
    fn : 1d ndarray of floats
        frequency vector
    bin_step : scalar, optional, default is 1000
        Frequency step in Hz
    
    Returns
    -------
    RAOQ : scalar
        Rao quadratic entropy  
        
    References
    ---------
    
    1. Zhao, Yueqin. "Rao's Quadratic Entropy and Some New Applications" (2010). Doctor of Philosophy (PhD) dissertation, Mathematics and Statistics, Old Dominion University, DOI: 10.25777/qgak-sf09

    Examples
    --------
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs)
    >>> S_power = maad.sound.avg_power_spectro(Sxx_power) 
    >>> maad.features.frequency_raoq(S_power, fn)
    0.10556621228886422
    
    """
    
    # be sure they are ndarray
    X = np.asarray(S_power)    

    #Convert into bins
    X_bins, bins = into_bins(X, fn, bin_min=fn[0], bin_max=fn[-1], 
                            bin_step=bin_step, axis=None) 
              
    # Compute Rao Quadratic Entropy
    RAOQ = _raoQ(X_bins,bins)
    
    return RAOQ

#=============================================================================    

def tfsd (Sxx, fn, tn, flim=(2000,8000), mode='thirdOctave', display=False):
    """
    Compute the Time frequency derivation index (tfsd) from a spectrogram.
        
    Parameters
    ----------
    Sxx : ndarray of floats
        matrix : Spectrogram  
    fn : vector
        frequency vector corresponding to the spectrogram
    tn : vector
        time vector corresponding to the spectrogram 
    flim : tupple (fmin, fmax), optional, default is (2000, 8000)
        Frequency band used to compute tfsd. 
    mode : string {'thirdOctave','Octave'}, default is thirdOctave  
        Select the way to transform the spectrogram with linear bands into 
        octave bands    
    display : boolean, optional, default is False
        Display the 1st and 2nd derivation of the spectrogram

    Returns
    -------    
    tfsd : scalar
        Time frequency derivation index
        
    Notes
    -----
    The higher the TFSD varies between 0 and 1, the greater the temporal 
    presence of avian or human vocalizations.  
    With the default configuration, a TFSD > 0.3 indicates a very important 
    presence time of the vocalizations in the signal. 
    The TFSD is always greater than 0.
       
    References 
    ----------
    .. [1] Aumond, P., Can, A., De Coensel, B., Botteldooren, D., Ribeiro, C., & Lavandier, C. (2017). 
    Modeling soundscape pleasantness using perceptual assessments and acoustic measurements 
    along paths in urban context. Acta Acustica united with Acustica,
    .. [2] Gontier, F., Lavandier, C., Aumond, P., Lagrange, M., & Petiot, J. F. (2019). 
    Estimation of the perceived time of presence of sources in urban acoustic environments 
    using deep learning techniques. Acta Acustica united with Acustica,
    
    Examples
    --------
    
    During the day
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs)
    >>> maad.features.tfsd(Sxx_power,fn, tn)  
    0.5002113200343906
    
    During the night
    
    >>> s, fs = maad.sound.load('../data/cold_forest_night.wav')
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs)
    >>> maad.features.tfsd(Sxx_power,fn, tn)  
    0.012818445992714088
    
    """
    # convert into 1/3 octave
    if mode == 'thirdOctave' : 
        x, fn_bin = linear_to_octave(Sxx, fn, thirdOctave=True)
    elif mode == 'Octave' : 
        x, fn_bin = linear_to_octave(Sxx, fn, thirdOctave=False)   

    # Derivation along the time axis, for each frequency bin
    GRADdt = diff(x, n=1, axis=1)
    # Derivation of the previously derivated matrix along the frequency axis 
    GRADdf = diff(GRADdt, n=1, axis=0)

    # select the bandwidth
    if flim is not None :
        GRADdf_select = GRADdf[index_bw(fn_bin[0:-1],bw=flim),]
    else :
        GRADdf_select = GRADdf    
    
    # calcul of the tfsdt : sum of the pseudo-gradient in the frequency bandwidth
    # which is normalized by the total sum of the pseudo-gradient
    tfsd =  sum(abs(GRADdf_select))/sum(abs(GRADdf)) 
    
    if display :
        
            extent=(tn[0], tn[-1], fn_bin[0], fn_bin[-1])
        
            fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
            # set the paramteers of the figure
            fig.set_facecolor('w')
            fig.set_edgecolor('k')
            fig.set_figheight(4)
            fig.set_figwidth (13)
                    
            # display image
            _im1 = ax1.imshow(power2dB(GRADdt), 
                              vmax = max(power2dB(GRADdt)), 
                              vmin = min(power2dB(GRADdt)),
                              interpolation='none', origin='lower', 
                              cmap='gray', extent=extent)
            plt.colorbar(_im1, ax=ax1)
            
            # set the parameters of the subplot
            ax1.set_title('Derivation along time axis')
            ax1.set_xlabel('Time [sec]')
            ax1.set_ylabel('Frequency [Hz]')   
            ax1.axis('tight') 
            
            # display image
            _im2 = ax2.imshow(power2dB(GRADdf), 
                              vmax = max(power2dB(GRADdf)), 
                              vmin = min(power2dB(GRADdf)),
                              interpolation='none', origin='lower', 
                              cmap='gray', extent=extent)
            plt.colorbar(_im2, ax=ax2)
       
            # set the parameters of the subplot
            ax2.set_title('Derivation along frequency axis')
            ax2.set_xlabel('Time [sec]')
            ax2.set_ylabel('Frequency [Hz]')
            ax2.axis('tight') 
         
            fig.tight_layout()
             
            # Display the figure now
            plt.show()
    
    return tfsd

#=============================================================================
def acoustic_gradient_index(Sxx, dt, order=1, norm='per_bin', display=False):
    """
    Compute the Acoustic Gradient Index (AGI) from a raw spectrogram.
    
    This index must be computed on a raw spectrogram (background noise must remain).
    
    Parameters
    ----------
    Sxx : ndarray of floats
        2d : Spectrogram 
    dt : float
        Time resolution in seconds. 
    norm : string, optional, default is 'per_bin'
        Determine if the AGI is normalized by the global meaian value 
        ('global' mode) or by the median value per frequency bin 
        ('per_bin')
        
    Returns
    -------
    AGI_xx : 2d ndarray of scalars
        Acoustic Gradient Index of the spectrogram
    AGI_per_bin : 1d ndarray of scalars
        AGI value for each frequency bin
        sum(AGI_xx,axis=1)  
    AGI_sum : scalar
        Sum of AGI value per frequency bin (Common definition)
        sum(AGI_per_bin)  
    AGI_mean ; scalar
        average AGI value per frequency bin (independant of the number of 
        frequency bin)
        mean(AGI_per_bin)
    
    Examples
    --------
    
    During the day
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs)
    >>> _, _, AGI_mean, _ = maad.features.acoustic_gradient_index(Sxx_power,tn[1]-tn[0])
    >>> AGI_mean
    5.026112548525072
    
    During the night
    
    >>> s, fs = maad.sound.load('../data/cold_forest_night.wav')
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs)
    >>> _, _, AGI_mean, _ = maad.features.acoustic_gradient_index(Sxx_power,tn[1]-tn[0])
    >>> AGI_mean
    1.45631461307782  
     
    """     
    # derivative (order = 1, 2, 3...)
    AGI_xx = abs(diff(Sxx, order, axis=1)) / (dt**order )
    
    if norm is not None :
        # Normalize the derivative by the median derivative which should 
        # correspond to the background (noise) derivative
        if norm =='per_bin':
            m = median(AGI_xx, axis=1)    
            m[m==0] = _MIN_    # Avoid dividing by zero value
            AGI_xx = AGI_xx/m[:,None]
        elif norm == 'global':
            m = median(AGI_xx) 
            if m==0: m = _MIN_ 
            AGI_xx = AGI_xx/m

    # mean per bin 
    AGI_per_bin = mean (AGI_xx,axis=1) 
    # Mean global
    AGI_mean = mean(AGI_per_bin) 
    # global sum
    AGI_sum = np.sum(AGI_per_bin) 

    # display full SPECTROGRAM in dB
    if display==True :
        
        fig4, ax4 = plt.subplots()
        # set the paramteers of the figure
        fig4.set_facecolor('w')
        fig4.set_edgecolor('k')
        fig4.set_figheight(4)
        fig4.set_figwidth (13)
                
        # display image
        _im = ax4.imshow(power2dB(Sxx**2), 
                         interpolation='none', origin='lower', 
                         vmin =20, vmax=70, cmap='gray')
        plt.colorbar(_im, ax=ax4)
 
        # set the parameters of the subplot
        ax4.set_title('Spectrogram')
        ax4.set_xlabel('Time [sec]')
        ax4.set_ylabel('Frequency [Hz]')
        ax4.axis('tight') 
     
        fig4.tight_layout()
         
        # Display the figure now
        plt.show()
        
    return AGI_xx, AGI_per_bin, AGI_mean, AGI_sum

#=============================================================================

def region_of_interest_index(Sxx_dB_noNoise, tn, fn, 
                          smooth_param1=1, mask_mode='relative', 
                          mask_param1=6, mask_param2=0.5, 
                          min_roi=25, max_roi=512*10000, 
                          display=False, **kwargs):
    """
    Compute an acoustic activity index based on the regions of interested detected on a spectrogram.
    
    The function first find regions of interest (ROI) and then compute the number and cover area 
    on the spectrogram.
    
    Parameters
    ----------
    Sxx_dB_noNoise : ndarray of floats
        Spectrogram without noise (i.e matrix of spectrum)
    tn : 1d ndarray of floats
        time vector (horizontal x-axis)
    fn : 1d ndarray of floats
        Frequency vector (vertical y-axis) 
    smooth_param1 : scalar, default is 1
        Standard deviation of the gaussian kernel used to smooth the image 
        The larger is the number, the smoother will be the image and the longer 
        it takes. Standard values should fall between 0.5 to 3 
    mask_mode : string in {'relative', 'absolute'}, optional, default is 'relative'
        if 'relative':
            Binarize an image based on a double relative threshold.  
            The values used for the thresholding depends on the values found 
            in the image. => relative threshold 
        if 'absolute' :
            Binarize an image based on a double relative threshold.  
            The values used for the thresholding are independent of the values 
            in the image => absolute threshold 
    mask_param1 : scalar, default is 6
        if 'relative' : bin_h
        if 'absolute' : bin_std
    mask_param2 : scalar, default is 0.5
        if 'relative' : bin_l
        if 'absolute' : bin_per
    min_roi, max_roi : scalars, optional, default : 9,  512*10000
        Define the minimum and the maximum area possible for an ROI. If None,  
        the minimum ROI area is 1 pixel and the maximum ROI area is the area of  
        the image     
    display : boolean, default is false
        plot graphs and spectrograms
    /*/*kwargs optional. This parameter is used by plt.plot and savefig functions 

    Returns
    -------    
    ROItotal : float
         Total number of ROIs found. The higher is the number of ROI, the higher
         is the acoustic abondance and/or richness expected
    ROIcover : float
        Percentage of spectrogram cover. The higher is the cover percentage, 
        the higher is the acoustic richness expected.
        
    Examples
    -------- 
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs)
    >>> Sxx_noNoise= maad.sound.median_equalizer(Sxx_power) 
    >>> Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)
    >>> ROItotal, ROIcover = maad.features.region_of_interest_index(Sxx_dB_noNoise, tn, fn, display=True)
    >>> print('The total number of ROIs found in the spectrogram is %2.0f' %ROItotal)
    The total number of ROIs found in the spectrogram is 265
    >>> print('The percentage of spectrogram covered by ROIs is %2.0f%%' %ROIcover)
    The percentage of spectrogram covered by ROIs is 12%
    
    """ 
    
    # extent
    kwargs.update({'extent':(tn[0], tn[-1], fn[0], fn[-1])})
    
    # Smooth the spectrogram in order to facilitate the creation of masks
    Sxx_dB_noNoise_smooth = smooth(Sxx_dB_noNoise, std=smooth_param1, 
                         display=display, savefig=None,**kwargs) 
    # binarization of the spectrogram to select part of the spectrogram with 
    # acoustic activity
    if mask_mode == 'relative' :
        im_mask = create_mask(Sxx_dB_noNoise_smooth,  
                              mode_bin = 'relative', bin_std=mask_param1, 
                              bin_per=mask_param2,
                              display=display, savefig=None, **kwargs) 
    elif  mask_mode == 'absolute' :   
        im_mask = create_mask(Sxx_dB_noNoise_smooth,  
                              mode_bin = 'absolute', bin_h=mask_param1, 
                              bin_l=mask_param2,
                              display=display, savefig=None, **kwargs)    
    else:
        raise TypeError ('mask_mode should be selected in {relative, absolute}')
            
    # get the mask with rois (im_rois) and the bounding box for each rois (rois_bbox) 
    # and an unique index for each rois => in the pandas dataframe rois
    im_rois, rois  = select_rois(im_mask,min_roi=9, 
                                 display= display, **kwargs)

    ##### Extract centroids features of each roi from the spectrogram in dB without noise 
    X = dB2power(Sxx_dB_noNoise)
    rois = format_features(rois, tn, fn) 
    centroid = centroid_features(X, rois, im_rois)
    
    if display :
        X = Sxx_dB_noNoise
        kwargs.update({'vmax':np.max(X)})
        kwargs.update({'vmin':np.min(X)})
        ax, fig = overlay_rois(X, rois, **kwargs)
 
#        dpi= 96
#        bbox_inches= 'tight'
#        format='png'
#        savefilename='_spectrogram_bounding_box'
#        filename = savefile+savefilename+'.'+format
#        print('\n''save figure : %s' %filename)
#        fig20.savefig(fname=filename, dpi=dpi, bbox_inches=bbox_inches,
#                    format=format)  

    #ROItotal
    ROItotal = len(centroid)
    
    ##### calcul the area of each roi
    # rectangular area (overestimation)
    area = (rois.max_y -rois.min_y) * (rois.max_x -rois.min_x)
    # size of im_rois => whole spectrogram
    x,y = im_rois.shape
    total_area = x*y
    # Pourcentage of ROI over the total area
    ROIcover = sum(area) / total_area *100
    
    return  ROItotal, ROIcover


#=============================================================================
def all_temporal_alpha_indices(s, fs, verbose=False, display=False, **kwargs):
    """
    Compute 16 temporal domain acoustic indices.

    Parameters
    ----------
    s : 1D array
        Audio to process (wav)
    fs : float
        Sampling frequency of the audio (Hz)
    verbose : boolean, default is False
        print indices on the default terminal
    display : boolean, default is False
        Display graphs
    \*\*kwargs : arguments for functions :
        - temporal_leq(s, fs, gain, Vadc, sensitivity, dBref, dt)
        - temporal_snr(s, mode, Nt) 
        - temporal_median(s, mode, Nt)
        - temporal_entropy(s, compatibility, mode, Nt)
        - temporal_activity (s,dB_threshold, mode, Nt)
        - temporal_events (s, fs, dB_threshold, rejectDuration, mode, Nt,display)
    
        For envelope
        mode : str, optional, default is "fast"
            Select the mode to compute the envelope of the audio waveform
            - "fast" : The sound is first divided into frames (2d) using the 
                function _wave2timeframes(s), then the max of each frame gives a 
                good approximation of the envelope.
            - "Hilbert" : estimation of the envelope from the Hilbert transform. 
                The method is slow
        Nt : integer, optional, default is 512
            Size of each frame. The largest, the highest is the approximation.
            
        For entropy
        compatibility : string {'QUT', 'seewave'}, default is 'QUT'
            Select the way to compute the temporal entropy.
                - QUT : entropy of the envelope²
                - seewave : entropy of the envelope
                
        For LEQt calculation
        gain : integer
            Total gain applied to the sound (preamplifer + amplifier)
        Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
            Maximal voltage (peak to peak) converted by the analog to digital convertor ADC    
        sensitivity : float, optional, default is -35 (dB/V)
            Sensitivity of the microphone
        dBref : integer, optional, default is 94 (dBSPL)
            Pressure sound level used for the calibration of the microphone 
            (usually 94dB, sometimes 114dB)
        dt : float, optional, default is 1 (second)
            Integration step to compute the Leq (Equivalent Continuous Sound level) 
        
        for audio activity and events
        dB_threshold : scalar, optional, default is 3dB
            data >Threshold is considered to be an event 
            if the length is > rejectLength
        rejectDuration : scalar, optional, default is None
            event shorter than rejectDuration are discarded
            duration is in s
   
    Returns
    -------
    df_temporal_indices: Panda dataframe
       Dataframe containing of the calculated audio indices : ZCR, MEANt, 
       VARt, SKEWt, KURTt, LEQt, BGNt, SNRt, MED, Ht, ACTtFraction, 
       ACTtCount, ACTtMean, EVNtFraction, EVNtMean, EVNtCount
           
    See also
    --------
    temporal_moments, temporal_events, temporal_activity, temporal_entropy, 
    temporal_median, temporal_leq, temporal_snr, zero_crossing_rate

    Examples
    --------
    >>> s, fs = maad.sound.load('../data/cold_forest_night.wav')
    >>> df_tempora_indices_NIGHT = maad.features.all_temporal_alpha_indices (s,fs)
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> df_temporal_indices_DAY = maad.features.all_temporal_alpha_indices (s,fs)
    
    Variation between night and day
    
    >>> var = abs(df_temporal_indices_DAY - df_temporal_indices_NIGHT)/df_temporal_indices_NIGHT*100
    >>> print('LEQt var night vs day: %2.2f %%' % var.LEQt)
    LEQt var : 29.66 %
    >>> print('Ht var night vs day: %2.2f %%' % var.Ht)
    Ht var : 2.33 %
    >>> print('MEANt var night vs day: %2.2f %%' % var.MEANt)
    MEANt var night vs day: 299.62 %
    >>> print('VARt var night vs day: %2.2f %%' % var.VARt)
    VARt var night vs day: 1664.02 %
    >>> print('EVNtFraction var night vs day: %2.2f %%' % var.EVNtFraction)
    EVNtFraction var night vs day: 98.48 %
    
    """
    
    #### get variables
    # Envelope => mode {'fast', 'hilbert"}, if 'fast', set Nt, number of point by frame 
    mode = kwargs.pop('mode','fast')
    Nt = kwargs.pop('Nt',512)
    
    # for entropy : compatibility {'QUT', 'seewave'}
    compatibility = kwargs.pop('compatibility','QUT') 
    
    # for LEQ : 
    gain = kwargs.pop('gain',42)
    Vadc = kwargs.pop('Vadc',2)
    dt = kwargs.pop('dt',1)
    sensitivity = kwargs.pop('sensitivity',-35)
    dBref = kwargs.pop('dBref',94)
    
    # for audio activity and events
    dB_threshold = kwargs.pop('dB_threshold',3)
    rejectDuration = kwargs.pop('rejectDuration',0.01)
    
    #### create a list
    df_temporal_indices=[] 
    
    """************************* Zero Crossing Rate ************ ***********""" 
    ZCR = zero_crossing_rate(s,fs)
    df_temporal_indices += [ZCR]
    if verbose :
        print("ZCR %2.5f" % ZCR)
        
    """**************************** 4 audio moments *************************""" 
    MEANt, VARt, SKEWt, KURTt = temporal_moments(s)
    df_temporal_indices += [MEANt, VARt, SKEWt, KURTt]
    if verbose :
        print("MEANt %2.5f" % MEANt)
        print("VARt %2.5f" % VARt)
        print("SKEWt %2.5f" % SKEWt)
        print("KURTt %2.5f" % KURTt)

    """********** total sound pressure level in temporal domain ***********""" 
    LEQt = temporal_leq(s, fs, gain, Vadc, sensitivity, dBref, dt)
    df_temporal_indices += [LEQt]
    if verbose :
        print("LEQt %2.5f" % LEQt)
    
    """************ Signal to noise Ratio and noise energy   *************"""
    _,BGNt,SNRt = temporal_snr(s, mode, Nt)  
    df_temporal_indices += [BGNt, SNRt]
    if verbose :
        print("SNRt %2.5f" % SNRt) 
        print("BGNt %2.5f" % BGNt)
    
    """*********************** median energy   ***************************"""
    MED =  temporal_median(s, mode, Nt)
    df_temporal_indices += [MED]
    if verbose :
        print("MED %2.5f" % MED)
    
    """*******************  energy concentration : entropy****************"""
    Ht =  temporal_entropy(s, compatibility, mode, Nt)
    df_temporal_indices += [Ht]
    if verbose :
        print("Ht %2.5f" % Ht)
    
    """**************************** Acoustic activity ********************"""
    """ ACT & EVN [TOWSEY] """
    ACTtFraction, ACTtCount, ACTtMean = temporal_activity (s,dB_threshold,
                                                        mode, Nt)
    df_temporal_indices += [ACTtFraction, ACTtCount, ACTtMean]
    if verbose :
        print("ACTtFraction %2.5f" % ACTtFraction)
        print("ACTtCount %2.5f" % ACTtCount)
        print("ACTtMean %2.5f" % ACTtMean)
    
    EVNtFraction, EVNtMean, EVNtCount, _ = temporal_events (s, fs, dB_threshold,
                                                         rejectDuration,
                                                         mode, Nt,
                                                         display=display)
    df_temporal_indices += [EVNtFraction, EVNtMean, EVNtCount]
    if verbose :    
        print("EVNtFraction %2.5f" % EVNtFraction)
        print("EVNtMean %2.5f" % EVNtMean)
        print("EVNtCount %2.5f" % EVNtCount)
    
    df_temporal_indices = pd.DataFrame([df_temporal_indices], 
                                    columns=['ZCR',
                                            'MEANt', 
                                            'VARt', 
                                            'SKEWt', 
                                            'KURTt',
                                            'LEQt',
                                            'BGNt', 
                                            'SNRt',
                                            'MED',
                                            'Ht',
                                            'ACTtFraction', 
                                            'ACTtCount', 
                                            'ACTtMean',
                                            'EVNtFraction', 
                                            'EVNtMean', 
                                            'EVNtCount'])
    return df_temporal_indices


def all_spectral_alpha_indices (Sxx_power, tn, fn,
                      flim_low=[0,1000], 
                      flim_mid=[1000,10000], 
                      flim_hi=[10000,20000], 
                      verbose=False, display=False, **kwargs):
    """
    Computes the acoustic indices in spectral (spectrum (1d) or spectrogram (2d)) domain.

    Parameters
    ----------
    Sxx_power : 2D array of floats
        Power spectrogram to process (taken directly from maad.sound.spectrogram)
    tn : 1d ndarray of floats
        time vector (horizontal x-axis)
    fn : 1d ndarray of floats
        Frequency vector (vertical y-axis)
    flim_low : tupple, optional, default is (0,1000)
        Low frequency band in Hz
    flim_mid : tupple, optional, default is (1000,10000)
        mid frequency band in Hz
    flim_hi : tupple, optional, default is (10000,20000)
        high frequency band in Hz        
    verbose : boolean, default is False
        print indices on the default terminal
    display : boolean, default is False
        Display graphs
    \*\*kwargs : arguments for functions :
                spectral_leq
                frequency_entropy
                soundscape_index
                bioacoustics_index
                acoustic_diversity_index
                acoustic_eveness_index
                spectral_cover
                spectral_activity
                spectral_events
                tfsd
                region_of_interest_index
               
        For soundscape_index, bioacoustics_index, acoustic_diversity_index, acoustic_eveness_index
        R_compatible : string, optional, default is "soundecology"
            if 'soundecology', the result is similar to the package SoundEcology in R 
            Otherwise, the result is specific to maad
                
        For LEQf calculation
        gain : integer
            Total gain applied to the sound (preamplifer + amplifier)
        Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
            Maximal voltage (peak to peak) converted by the analog to digital convertor ADC    
        sensitivity : float, optional, default is -35 (dB/V)
            Sensitivity of the microphone
        dBref : integer, optional, default is 94 (dBSPL)
            Pressure sound level used for the calibration of the microphone 
            (usually 94dB, sometimes 114dB)
        pRef : Sound pressure reference in the medium (air : 20e-6, water : 1e-6)
        
        for spectral activity and events, ADI, AEI
        dB_threshold : scalar, optional, default is 3dB
            data >Threshold is considered to be an event 
            if the length is > rejectLength
            
        for spectral activity and events
        rejectDuration : scalar, optional, default is None
            event shorter than rejectDuration are discarded
            duration is in s
            
        for Roi
        smooth_param1 : scalar, default is 1
            Standard deviation of the gaussian kernel used to smooth the image 
            The larger is the number, the smoother will be the image and the longer 
            it takes. Standard values should fall between 0.5 to 3 
        mask_mode : string in {'relative', 'absolute'}, optional, default is 'relative'
            if 'relative':
                Binarize an image based on a double relative threshold.  
                The values used for the thresholding depends on the values found 
                in the image. => relative threshold 
            if 'absolute' :
                Binarize an image based on a double relative threshold.  
                The values used for the thresholding are independent of the values 
                in the image => absolute threshold 
        mask_param1 : scalar, default is 6
            if 'relative' : bin_h
            if 'absolute' : bin_std
        mask_param2 : scalar, default is 0.5
            if 'relative' : bin_l
            if 'absolute' : bin_per
        min_roi, max_roi : scalars, optional, default : 9,  512*10000
            Define the minimum and the maximum area possible for an ROI. If None,  
            the minimum ROI area is 1 pixel and the maximum ROI area is the area of  
            the image     
        
        for ADI, AEI, RAOQ
        bin_step : scalar, optional, default is 500
            Frequency step in Hz        
  
    Returns
    -------
    df_spectral_indices: Panda dataframe
        Dataframe containing of the calculated spectral indices :
    df_per_bin_indices : Panda dataframe
        Dataframe containing of the calculated spectral indices  per frequency
        bin :
           
    See Also
    --------
    number_of_peaks, spectral_leq, spectral_snr, frequency_entropy, 
    spectral_entropy, acoustic_complexity_index, soundscape_index, soundscape_index,
    roughness, acoustic_diversity_index, acoustic_eveness_index, spectral_cover, 
    spectral_activity, spectral_events, tfsd, more_entropy, frequency_raoq, 
    acoustic_gradient_index, region_of_interest_index

    Examples
    --------    
    Spectral indices on a daylight recording
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power,tn,fn,ext = maad.sound.spectrogram (s, fs)  
    >>> df_spectral_indices_DAY, _ = maad.features.all_spectral_alpha_indices(Sxx_power,tn,fn,display=True, extent=ext)
    
    Spectral indices on a night recording
    
    >>> s, fs = maad.sound.load('../data/cold_forest_night.wav')
    >>> Sxx_power,tn,fn,ext = maad.sound.spectrogram (s, fs)  
    >>> df_spectral_indices_NIGHT, _ = maad.features.all_spectral_alpha_indices(Sxx_power,tn,fn,display=True)
    
    Variation between night and day
    
    >>> var = abs(df_spectral_indices_DAY - df_spectral_indices_NIGHT)/df_spectral_indices_NIGHT*100
    >>> print('LEQf var night vs day: %2.2f %%' % var.LEQf)
    LEQf var night vs day: 34.94 %
    >>> print('Hf var night vs day: %2.2f %%' % var.Hf)
    Hf var night vs day: 105.61 %
    >>> print('ACI var night vs day: %2.2f %%' % var.ACI)
    ACI var night vs day: 3.39 %
    >>> print('AGI var night vs day: %2.2f %%' % var.AGI)
    AGI var night vs day: 20.50 %
    >>> print('ROItotal var night vs day: %2.2f %%' % var.ROItotal)
    ROItotal var night vs day: 248.68 %
    
    """
    
    # extent
    kwargs.update({'extent':(tn[0], tn[-1], fn[0], fn[-1])})
    
    #### get variables  
    R_compatible = kwargs.pop('R_compatible','soundecology') 
    
    # for LEQ : 
    gain = kwargs.pop('gain',42)
    Vadc = kwargs.pop('Vadc',2)
    sensitivity = kwargs.pop('sensitivity',-35)
    dBref = kwargs.pop('dBref',94)
    pRef = kwargs.pop('pRef',20e-6)
    
    # for audio activity and events
    dB_threshold = kwargs.pop('dB_threshold',3)
    rejectDuration = kwargs.pop('rejectDuration',None) # if None => 3 pixels
    
    ### for Roi
    min_roi_area    = kwargs.pop('min_roi_area',None) # if None =>  30ms * 100Hz
    smooth_param1   = kwargs.pop('smooth_param1',1)
    mask_mode       = kwargs.pop('mask_mode','relative')
    mask_param1     = kwargs.pop('mask_param1',6)
    mask_param2     = kwargs.pop('mask_param2',0.5)
    
    ### for ADI, AEI, RAOQ
    bin_step = kwargs.pop('bin_step',1000) # in Hz
    
    #### create a list
    df_spectral_indices=[] 
    df_per_bin_indices=[] 
    
    ### for flim to be ndarray
    flim_low = np.asarray(flim_low)
    flim_mid = np.asarray(flim_mid)
    flim_hi = np.asarray(flim_hi)
        
    #### Prepare different spectrograms and spectrums
    # amplitude spectrogram
    Sxx_amplitude = sqrt(Sxx_power)
    # mean amplitude spectrum
    S_amplitude = avg_amplitude_spectro(Sxx_amplitude)
    # mean power spectrum
    S_power = avg_power_spectro(Sxx_power)
    
    """************************* Long term spectrogram *********************"""
    # mean power spectrum => for long term spectrogram (LTS)
    LTS = avg_power_spectro(Sxx_power)
    df_per_bin_indices +=[fn.tolist()]
    df_per_bin_indices +=[LTS.tolist()]
    
    """**************************** 4 spectrum moments *********************""" 
    MEANf, VARf, SKEWf, KURTf = spectral_moments(S_amplitude)
    df_spectral_indices += [MEANf, VARf, SKEWf, KURTf]
    if verbose :
        print("MEANf %2.5f" % MEANf)
        print("VARf %2.5f" % VARf)
        print("SKEWf %2.5f" % SKEWf)
        print("KURTf %2.5f" % KURTf)
     
    """*********************** 4 audio moments per bin ********************""" 
    MEANt_per_bin, VARt_per_bin, SKEWt_per_bin, KURTt_per_bin = spectral_moments(Sxx_amplitude, axis=1) 
    MEANt_per_bin = np.asarray(MEANt_per_bin).tolist()
    VARt_per_bin = np.asarray(VARt_per_bin).tolist()
    SKEWt_per_bin = np.asarray(SKEWt_per_bin).tolist()
    KURTt_per_bin = np.asarray(KURTt_per_bin).tolist()
    df_per_bin_indices += [MEANt_per_bin,VARt_per_bin,
                           SKEWt_per_bin,KURTt_per_bin]
    """**************************** Number of peaks ************************"""
    NBPEAKS = number_of_peaks(S_amplitude,fn,display=display)
    df_spectral_indices += [NBPEAKS]
    if verbose :
        print("NBPEAKS %2.5f" % NBPEAKS)
    
    """********* total sound pressure level in frequency domain ************"""
    LEQf, LEQf_per_bin = spectral_leq(Sxx_power, gain, Vadc, sensitivity, dBref, pRef)
    df_spectral_indices += [LEQf]
    df_per_bin_indices += [np.asarray(LEQf_per_bin).tolist()]
    if verbose :
        print("LEQf %2.5f" % LEQf)
    
    """************ Signal to noise Ratio and noise energy   *************"""
    """ SNRf [TOWSEY] """
    ENRf, BGNf, SNRf, ENRf_per_bin, BGNf_per_bin, SNRf_per_bin = spectral_snr(Sxx_power)
    df_spectral_indices += [ENRf, BGNf, SNRf]
    df_per_bin_indices += [np.asarray(ENRf_per_bin).tolist(),
                           np.asarray(BGNf_per_bin).tolist(),
                           np.asarray(SNRf_per_bin).tolist()]
    if verbose :
        print("ENRf %2.5f" % ENRf)
        print("BGNf %2.5f" % BGNf)
        print("SNRf %2.5f" % SNRf)

    """*******************  energy concentration : entropy ****************"""
    Hf, Ht_per_bin = frequency_entropy(Sxx_power, compatibility="QUT")
    df_spectral_indices += [Hf] 
    df_per_bin_indices += [np.asarray(Ht_per_bin).tolist()]
    if verbose :
        print("Hf %2.5f" % Hf)

    """*********************** Remove stationnary noise ********************"""       
    #### Use median_equalizer function as it is fast reliable
    Sxx_power_noNoise = median_equalizer(Sxx_power, display=display, **kwargs)    
    
    #### Convert into dB
    Sxx_dB_noNoise = power2dB(Sxx_power_noNoise)

    """******** Spectral indices from Spectrum (Amplitude or Energy) *******"""  
    """ EAS, ECU, ECV, EPS, KURT, SKEW [TOWSEY]  """
    #### Does not take into account low frequencies.
    EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW = spectral_entropy (Sxx_power_noNoise,
                                                               fn,
                                                               flim=(flim_mid[0],flim_hi[1]),
                                                               display=display)
    df_spectral_indices += [EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW] 
    if verbose :
        print("EAS %2.5f" % EAS)
        print("ECU %2.5f" % ECU)
        print("ECV %2.5f" % ECV)
        print("EPS %2.5f" % EPS)
        print("EPS_KURT %2.5f" % EPS_KURT)
        print("EPS_SKEW %2.5f" % EPS_SKEW)
    
    """=============================================================
    ECOLOGICAL INDICES :
            ACI
            NDSI 
            rBA 
            Bioacoustics Index
    ============================================================="""
    
    #### Acoustic complexity index => 1st derivative of the spectrogram
    """ ACI """
    _,ACI_per_bin,ACI_sum = acoustic_complexity_index(Sxx_amplitude)
    ACI=ACI_sum
    df_spectral_indices += [ACI]
    df_per_bin_indices += [np.asarray(ACI_per_bin).tolist()]
    if verbose :
        print("ACI {seewave} %2.5f" %ACI)

    #### energy repartition in the frequency bins
    """ NDSI & rBA """
    NDSI, rBA, AnthroEnergy, BioEnergy = soundscape_index(Sxx_power, fn, 
                                                         flim_bioPh=flim_mid,
                                                         flim_antroPh=flim_low,
                                                         R_compatible=R_compatible) 
    df_spectral_indices += [NDSI, rBA, AnthroEnergy, BioEnergy]
    if verbose :                                         
        if R_compatible == 'soundecology' :
            print("NDSI {soundecology} %2.5f" %NDSI)
        else :
            print("NDSI {seewave} %2.5f" %NDSI)
   
    ###### Bioacoustics Index : the calculation in R from soundecology is weird...
    """ BI """
    BI = bioacoustics_index(Sxx_amplitude, fn, 
                           flim=flim_mid, R_compatible=R_compatible) 
    df_spectral_indices += [BI]
    if verbose :
        if R_compatible == 'soundecology' :
            print("BI {SoundEcology} %2.5f" %BI)
        else :
            print("BI {MAAD} %2.5f" %BI)
    
    #### roughness
    """ ROU """
    ROU_per_bin = roughness(Sxx_amplitude, norm=None, axis=1)
    ROU = np.sum(ROU_per_bin) 
    df_spectral_indices += [ROU]
    df_per_bin_indices += [np.asarray(ROU_per_bin).tolist()]
    if verbose :
        print("roughness %2.2f" % ROU)    
    
    """*********** Spectral indices from the decibel spectrogram ***********"""
    #### Score
    """ ADI & AEI """ 
    """ 
        COMMENT :
                - threshold : -50dB when norm by the max (as soundecology)
                              6dB if PSDxxdB_SansNoise
    """  
    ADI = acoustic_diversity_index(Sxx_amplitude, fn, fmin=flim_low[0], 
                                 fmax=flim_mid[1], bin_step=bin_step, 
                                 dB_threshold=-50, index="shannon") 
    AEI = acoustic_eveness_index(Sxx_amplitude, fn, fmin=flim_low[0], 
                               fmax=flim_mid[1], bin_step=bin_step, 
                               dB_threshold=-50) 
    df_spectral_indices += [ADI, AEI]
    if verbose :
        print("ADI %2.5f" %ADI)
        print("AEI %2.5f" %AEI)
               
    """************************** SPECTRAL COVER ***************************"""
    #### frequency cover 
    """ LFC, MFC, HFC [TOWSEY] """
    LFC, MFC, HFC = spectral_cover (Sxx_dB_noNoise, fn,dB_threshold=dB_threshold, 
                                    flim_LF=flim_low,flim_MF=flim_mid,flim_HF=flim_hi)
    df_spectral_indices += [LFC, MFC, HFC]
    if verbose :
        print("LFC %2.5f" %LFC)
        print("MFC %2.5f" %MFC)
        print("HFC %2.5f" %HFC)
    
    """**************************** Activity *******************************"""
    # Time resolution (in s)
    DELTA_T = tn[1]-tn[0]
    
    if rejectDuration is None :
        rejectDuration = DELTA_T * 3
    
    X = Sxx_dB_noNoise
    ACTspFract, ACTspCount, ACTspMean = spectral_activity (X, dB_threshold=dB_threshold)
    ACTspFract_avg = np.mean(ACTspFract)
    ACTspCount_avg = np.mean(ACTspCount)
    df_spectral_indices += [ACTspFract_avg, ACTspCount_avg, ACTspMean]
    df_per_bin_indices += [np.asarray(ACTspFract).tolist(),
                           np.asarray(ACTspCount).tolist()]
    if verbose :
        print("ACTspFract %2.5f" %ACTspFract_avg)
        print("ACTspCount %2.5f" %ACTspCount_avg)
        print("ACTspMean %2.5f" %ACTspMean)

    EVNspFract, EVNspMean, EVNspCount, _ = spectral_events (X, 
                                                            dt=DELTA_T,
                                                            dB_threshold=dB_threshold,
                                                            rejectDuration=rejectDuration,
                                                            display=display,
                                                            **kwargs)
    EVNspFract_avg = np.mean(EVNspFract)
    EVNspMean_avg = np.mean(EVNspMean)
    EVNspCount_avg = np.mean(EVNspCount)
    df_spectral_indices += [EVNspFract_avg, EVNspMean_avg, EVNspCount_avg]
    df_per_bin_indices += [np.asarray(EVNspFract).tolist(),
                           np.asarray(EVNspMean).tolist(),
                           np.asarray(EVNspCount).tolist()]
    if verbose :
        print("EVNspFract %2.5f" %mean(EVNspFract))
        print("EVNspMean %2.5f" %mean(EVNspMean))
        print("EVNspCount %2.5f" %mean(EVNspCount))
          
    """**************************** New indices*****************************""" 
    """ TFSD """
    # compute TFSD with mode = ThirdOctave and flim
    TFSD= tfsd(Sxx_amplitude,fn,tn,flim=flim_mid,mode='thirdOctave')
    df_spectral_indices += [TFSD]
    if verbose :
        print("TFSD %2.5f" % TFSD)
    
    """ More entropy"""
    X = S_power
    H_Havrda, H_Renyi, H_pairedShannon, H_gamma, H_GiniSimpson = more_entropy(X, order=3)
    df_spectral_indices += [H_Havrda, H_Renyi, H_pairedShannon, H_gamma, H_GiniSimpson]
    if verbose :
        print("H_Havrda %2.2f" % H_Havrda)
        print("H_Renyi %2.2f" % H_Renyi)
        print("H_pairedShannon %2.2f" % H_pairedShannon)
        print("H_gamma %2.2f" % H_gamma)
        print("H_GiniSimpson %2.2f" % H_GiniSimpson)  

    """ RAOQ """
    X = S_power
    RAOQ = frequency_raoq(X, fn, bin_step=bin_step) 
    df_spectral_indices += [RAOQ]
    if verbose :
        print("RAOQ %2.2f" % RAOQ)
    
    #### Acoustic gradient index => real 1st derivative of the spectrogram
    """ AGI """
    # Time resolution (in s)
    DELTA_T = tn[1]-tn[0]
    X = Sxx_amplitude
    _, AGI_per_bin, AGI, _ = acoustic_gradient_index(X, dt=DELTA_T, 
                                                   order=1, norm='per_bin')
    df_spectral_indices += [AGI]
    df_per_bin_indices += [np.asarray(AGI_per_bin).tolist()]
    if verbose :
        print("AGI %2.3f" % AGI)
    
    """ ROI index """
    # Time resolution (in s)
    DELTA_T = tn[1]-tn[0]
    # Frequency resolution (in Hz)
    DELTA_F = fn[1]-fn[0]
    # Minimum time duration of an event (in s)
    MIN_EVENT_DUR = 30e-3
    # Minimum frequency bandwidth (in Hz)
    MIN_FREQ_BW = 100
    # Min Region Of Interest ROI
    if min_roi_area is None :
        min_roi_area = int(MIN_EVENT_DUR/DELTA_T * MIN_FREQ_BW / DELTA_F)
    ROItotal, ROIcover = region_of_interest_index(Sxx_dB_noNoise, 
                                               tn, fn, 
                                               smooth_param1, 
                                               mask_mode,
                                               mask_param1, 
                                               mask_param2, 
                                               min_roi=min_roi_area, 
                                               display=display)
    df_spectral_indices += [ROItotal, ROIcover]
    if verbose :
        print("ROItotal %2.3f" % ROItotal)
        print("ROIcover %2.3f" % ROIcover)
        
    df_spectral_indices = pd.DataFrame([df_spectral_indices], 
                                    columns=['MEANf', 
                                             'VARf', 
                                             'SKEWf', 
                                             'KURTf', 
                                             'NBPEAKS', 
                                             'LEQf', 
                                             'ENRf', 
                                             'BGNf', 
                                             'SNRf',
                                             'Hf', 
                                             'EAS',
                                             'ECU',
                                             'ECV',
                                             'EPS',
                                             'EPS_KURT',
                                             'EPS_SKEW',
                                             'ACI',
                                             'NDSI',
                                             'rBA',
                                             'AnthroEnergy',
                                             'BioEnergy',
                                             'BI',
                                             'ROU',
                                             'ADI',
                                             'AEI',
                                             'LFC',
                                             'MFC',
                                             'HFC',
                                             'ACTspFract',
                                             'ACTspCount',
                                             'ACTspMean', 
                                             'EVNspFract',
                                             'EVNspMean',
                                             'EVNspCount',
                                             'TFSD',
                                             'H_Havrda',
                                             'H_Renyi',
                                             'H_pairedShannon', 
                                             'H_gamma', 
                                             'H_GiniSimpson',
                                             'RAOQ',
                                             'AGI',
                                             'ROItotal',
                                             'ROIcover'])
        
    df_per_bin_indices = pd.DataFrame([df_per_bin_indices], 
                                    columns=['frequencies',
                                            'LTS',
                                            'MEANt_per_bin',
                                            'VARt_per_bin', 
                                            'SKEWt_per_bin',
                                            'KURTt_per_bin',
                                            'LEQf_per_bin',
                                            'ENRf_per_bin',
                                            'BGNf_per_bin',
                                            'SNRf_per_bin',
                                            'Ht_per_bin',
                                            'ACI_per_bin',
                                            'ROU_per_bin',
                                            'ACTspFract_per_bin',
                                            'ACTspCount_per_bin',
                                            'EVNspFract_per_bin',
                                            'EVNspMean_per_bin',
                                            'EVNspCount_per_bin',
                                            'AGI_per_bin'])
                                    
    return df_spectral_indices, df_per_bin_indices

