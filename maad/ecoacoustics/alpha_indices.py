#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Created on Wed Oct 24 11:56:28 2018
    Alpha indices used in ecoacoustics"""
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>        
#
# License: New BSD License

"""****************************************************************************
# -------------------       Load modules            ---------------------------
****************************************************************************"""

#### Import external modules
import numpy as np 
from numpy import sum, log,  min, max, abs, mean, sqrt, diff
import numbers
from scipy.signal import hilbert
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.stats import rankdata

import matplotlib.pyplot as plt

#### Importation from internal modules
from ..util import rle, index_bw, linear_scale, dB2linear, linear2dB

# min value
import sys
_MIN_ = sys.float_info.min

# =============================================================================
# List of functions
# =============================================================================

 
#=============================================================================
def wave2frames (wave, N=512):
    """
    Reshape a sound waveform (ie vector) into a serie of frames (ie matrix) of
    length N
    
    Parameters
    ----------
    wave : 1d ndarray of floats (already divided by the number of bits)
        Vector containing the sound waveform 

    N : int, optional, default is 512
        Number of points per frame
                
    Returns
    -------
    timeframes : 2d ndarray of floats
        Matrix containing K frames (row) with N points (column), K*N <= length (wave)
    """
    # transform wave into array
    wave = np.asarray(wave)
    # compute the number of frames
    K = len(wave)//N
    # Reshape the waveform (ie vector) into a serie of frames (ie 2D matrix)
    timeframes = wave[0:K*N].reshape(-1,N).transpose()
    return timeframes

#=============================================================================
def intoBins (x, an, bin_step, axis=0, bin_min=None, bin_max=None, display=False):
    """ 
    Transform a vector or a matrix into bins 
    
    Parameters
    ----------
    x : array-like
        1D or 2D array.
    an :1d ndarray of floats 
        Vector containing the positions of each value. 
        In case of 2D matrix, this vector corresponds to the horizontal (row)
        or vertical (columns) units
    bin_step : scalar
        Determine the width of each bin.
    axis : integer, optional, default is 0
        Determine  along which axis the transformation is done.
        In case of matrix :
            axis = 0 => transformation is done on column
            axis = 1 => transformation is done on row 
    bin_min : scalar, optional, default is None
        This minimum value corresponds to the start of the first bin. 
        By default, the minimum value is the first value of an.
    bin_max : scalar, optional, default is None
        This maximum value corresponds to end of the last bin. 
        By default, the maximum value is the last value of an.   
    display : boolean, optional, defualt is False
        Display the result of the tranformation : an histogram
        In case of matrix, the mean histogram is shown.
        
    Returns
    -------
    xbins :  array-like
        1D or 2D array which correspond to the data after being transformed into bins
    bin : 1d ndarray of floats 
        Vector containing the positions of each bin
    """    
    
    # Test if the bin_step is larger than the resolution of an
    if bin_step < (an[1]-an[0]):
        raise Exception('WARNING: bin step must be largero or equal than the actual resolution of x')

    # In case the limits of the bin are not set
    if bin_min == None:
        bin_min = an[0]
    if bin_max == None :
        bin_max = an[-1]
    
    # Creation of the bins
    bins = np.arange(bin_min,bin_max+bin_step,bin_step)
    
    # select the indices corresponding to the frequency bins range
    b0 = bins[0]
    xbin = []
    s = []
    for index, b in enumerate(bins[1:]):
        indices = (an>=b0)*(an<b) 
        s.append(sum(indices))
        if axis==0:
            xbin.append(mean(x[indices,:],axis=axis))
        elif axis==1:
            xbin.append(mean(x[:,indices],axis=axis))
        b0 = b
      
    xbin = np.asarray(xbin) * mean(s)
    bins = bins[0:-1]
    
    # Display
    if display:
        plt.figure()
        # if xbin is a vector
        if xbin.ndim ==1:
            plt.plot(an,x)
            plt.bar(bins,xbin, bin_step*0.75, alpha=0.5, align='edge')
        else:
            # if xbin is a matrix
            if axis==0 : axis=1
            elif axis==1 : axis=0
            plt.plot(an,mean(x,axis=axis))
            plt.bar(bins,mean(xbin,axis=1), bin_step*0.75, alpha=0.5, align='edge')
    
    return xbin, bins

#=============================================================================
def skewness (x, axis=0):
    """
    Calcul the skewness (asymetry) of a signal x
    
    Parameters
    ----------
    x : ndarray of floats 
        1d signal or 2d matrix
        
    axis : integer, optional, default is 0
        select the axis to compute the kurtosis
                            
    Returns
    -------    
    ku : float or ndarray of floats
        skewness of x 
        if x is a 1d vector => single value
        if x is a 2d matrix => array of values corresponding to the number of
                               points in the other axis
        
    """
    if isinstance(x, (np.ndarray)) == True:
        Nf = x.shape[axis]
        mean_x =  mean(x, axis=axis)
        std_x = np.std(x, axis=axis)
        z = x - mean_x
        sk = (sum(z**3)/(Nf-1))/std_x**3
    else:
        print ("WARNING: type of x must be ndarray") 
        sk = None
       
    return sk

#=============================================================================
def kurtosis (x, axis=0):
    """
    Calcul the kurtosis (tailedness or curved or arching) of a signal x
    
    Parameters
    ----------
    x : ndarray of floats 
        1d signal or 2d matrix
        
    axis : integer, optional, default is 0
        select the axis to compute the kurtosis
                            
    Returns
    -------    
    ku : float or ndarray of floats
        kurtosis of x 
        if x is a 1d vector => single value
        if x is a 2d matrix => array of values corresponding to the number of
                               points in the other axis
        
    """
    if isinstance(x, (np.ndarray)) == True:
        Nf = x.shape[axis]
        mean_x =  mean(x, axis=axis)
        std_x = np.std(x, axis=axis)
        z = x - mean_x
        ku = (sum(z**4)/(Nf-1))/std_x**4
    else:
        print ("WARNING: type of x must be ndarray") 
        ku = None
       
    return ku

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
        'global' : normalize by the maximum value in the vector or matrix
        'per_axis' : normalize by the maximum value found along each axis

    axis : int, optional, default is 0
        select the axis where the second derivation is computed
        if x is a vector, axis=0
        if x is a 2d ndarray, axis=0 => rows, axis=1 => columns
                
    Returns
    -------
    y : float or ndarray of floats

    Reference
    ---------
    Described in [Ramsay JO, Silverman BW (2005) Functional data analysis.]
    Ported from SEEWAVE R Package
    """      
    
    if norm is not None:
        if norm == 'per_axis' :
            m = max(x, axis=axis) 
            m[m==0] = _MIN_    # Avoid dividing by zero value
            if axis==0:
                x = x/m[None,:]
            elif axis==1:
                x = x/m[:,None]
        elif norm == 'global' :
            m = max(x) 
            if m==0 : m = _MIN_    # Avoid dividing by zero value
            x = x/m 
            
    deriv2 = diff(x, 2, axis=axis)
    r = sum(deriv2**2, axis=axis)
    
    return r

#=============================================================================
def entropy (datain, axis=0):
    """
    Computes the entropy of a vector or matrix datain (i.e. waveform, spectrum...)    
    
    Parameters
    ----------
    datain : ndarray of floats
        datain is a vector (1d) or a matrix (2d)

    axis : int, optional, default is 0
        select the axis where the entropy is computed
        if datain is a vector, axis=0
        if datain is a 2d ndarray, axis=0 => rows, axis=1 => columns
                
    Returns
    -------
    H : float or ndarray of floats

    """
    if isinstance(datain, (np.ndarray)) == True:
        if datain.ndim > axis:
            if datain.shape[axis] == 0: 
                print ("WARNING: x is empty") 
                H = None 
            elif datain.shape[axis] == 1:
                H = 0 # null entropy
            elif sum(datain) == 0:
                H = 0 # null entropy
            else:
                # if datain contains negative values -> rescale the signal between 
                # between posSitive values (for example (0,1))
                if np.min(datain)<0:
                    datain = linear_scale(datain,minval=0,maxval=1)
                # length of datain along axis
                n = datain.shape[axis]
                # Tranform the signal into a Probability mass function (pmf)
                # Sum(pmf) = 1
                if axis == 0 :
                    pmf = datain/sum(datain,axis)
                elif axis == 1 : 
                    pmf = (datain.transpose()/sum(datain,axis)).transpose()
                pmf[pmf==0] = _MIN_
                #normalized by the length : H=>[0,1]
                H = -sum(pmf*log(pmf),axis)/log(n)
        else:
            print ("WARNING :axis is greater than the dimension of the array")    
            H = None 
    else:
        print ("WARNING: type of datain must be ndarray")   
        H = None 

    return H

#=============================================================================
def envelope (wave, mode='fast', N=512):
    """
    Calcul the envelope of a sound (1d) 
    The sound is first divided into frames (2d) using the function 
    wave2timeframes(wave), then the max of each frame gives a good approximation
    of the envelope.
    
    Parameters
    ----------
    wave : ndarray of floats 
        1d : sound waveform
        
    mode : str, optional, default is "fast"
        - "fast" : The sound is first divided into frames (2d) using the 
            function wave2timeframes(wave), then the max of each frame gives a 
            good approximation of the envelope.
        - "Hilbert" : estimation of the envelope from the Hilbert transform. 
            The method is slow
    
    N : integer, optional, default is 512
        Size of each frame. The largest, the highest is the approximation.
                  
    Returns
    -------    
    env : ndarray of floats
        Envelope of the sound (1d) 
        
    Reference
    ---------
    Fast calculation is inspired by the work of Towsey.
    """
    if mode == 'fast' :
        # Envelope : take the median (instead of max as Towsey) of each frame
        # median gives more accurate result than max
        frames = wave2frames(wave, N=N)
        env = max(abs(frames),0) 
    elif mode =='hilbert' :
        # Compute the hilbert transform of the waveform and take the norm 
        # (magnitude) 
        env = np.abs(hilbert(wave))  
    else:
        print ("WARNING : choose a mode between 'fast' and 'hilbert'")
        
    return env


#=============================================================================
def score (x, threshold, axis=0):
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

#=============================================================================
def gini(x, corr=False):
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
        
    Reference
    ---------
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

#=============================================================================
def shannonEntropy(datain, axis=0):
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
    Hs = entropy(datain, axis=axis) * log(n)
    return Hs

#=============================================================================
def acousticRichnessIndex (Ht_array, M_array):
    """
    Acoustic richness index : AR
    
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
        
    Reference
    ---------
    Described in [Depraetere & al. 2012]
    Ported from SEEWAVE R package
    """    
    if len(Ht_array) != len(M_array) : 
        print ("warning : Ht_array and M_array must have the same length")
    
    AR = rankdata(Ht_array) * rankdata(M_array) / len(Ht_array)**2
    
    return AR

#=============================================================================
def acousticComplexityIndex(Sxx, norm ='global'):
    
    """
    Acoustic Complexity Index : ACI
    
    Parameters
    ----------
    Sxx : ndarray of floats
        2d : Spectrogram (i.e matrix of spectrum)
    
    norm : string, optional, default is 'global'
        Determine if the ACI is normalized by the sum on the whole frequencies
        ('global' mode) or by the sum of frequency bin per frequency bin 
        ('per_bin')

    Returns
    -------    
    ACI_xx : 2d ndarray of scalars
        Acoustic Complexity Index of the spectrogram
    
    ACI_per_bin : 1d ndarray of scalars
        ACI value for each frequency bin
        sum(ACI_xx,axis=1)
        
    ACI_sum : scalar
        Sum of ACI value per frequency bin (Common definition)
        sum(ACI_per_bin)
        
    ACI_mean ; scalar
        
        !!! pas de sens car non independant de la résolution freq et temporelle
        !!! Seulement sum donne un résultat independant de N (pour la FFT)  
        !!! et donc de df et dt
        
    Reference
    ---------
    Pieretti N, Farina A, Morri FD (2011) A new methodology to infer the singing 
    activity of an avian community: the Acoustic Complexity Index (ACI). 
    Ecological Indicators, 11, 868-873.
    
    Ported from the Seewave R package.
    
    !!!!! in Seewave, the result is the sum of the ACI per bin.
    
    """   
    if norm == 'per_bin':
        ACI_xx = ((abs(diff(Sxx,1)).transpose())/(sum(Sxx,1)).transpose()).transpose()
    elif norm == 'global':
        ACI_xx = (abs(diff(Sxx,1))/sum(Sxx))

    ACI_per_bin = sum(ACI_xx,axis=1)
    ACI_sum = sum(ACI_per_bin)
    
    return ACI_xx, ACI_per_bin, ACI_sum, 

def surfaceRoughness (Sxx, norm ='global'):
    
    """
    Surface Roughness 
    see wikipedia : https://en.wikipedia.org/wiki/Surface_roughness
    
    Parameters
    ----------
    Sxx : ndarray of floats
        2d : Spectrogram (i.e matrix of spectrum)
    
    norm : string, optional, default is 'global'
        Determine if the ROUGHNESS is normalized by the sum on the whole frequencies
        ('global' mode) or by the sum of frequency bin per frequency bin 
        ('per_bin')

    Returns
    -------        
    Ra_per_bin : 1d ndarray of scalars
        Arithmetical mean deviation from the mean line (global or per frequency bin)
        => ROUGHNESS value for each frequency bin
        
    Ra : scalar
        Arithmetical mean deviation from the mean line [mean (Ra_per_bin)]
        => mean ROUGHNESS value over Sxx 
        
    Rq_per_bin : 1d ndarray of scalars
        Root mean squared of deviation from the mean line (global or per frequency bin)
        => RMS ROUGHNESS value for each frequency bin
        
    Rq : scalar
        Root mean squared of deviation from the mean line  [mean (Rq_per_bin)]
        => RMS ROUGHNESS value over Sxx 
    """    
    if norm == 'per_bin':
        m = mean(Sxx, axis=1)
        y = Sxx-m[..., np.newaxis]
        
    elif norm == 'global':
        m = mean(Sxx)
        y = Sxx-m

    # Arithmetic mean deviation
    Ra_per_bin = mean(abs(y), axis=1)
    Ra = mean(Ra_per_bin)

    Rq_per_bin = sqrt(mean(y**2, axis=1))
    Rq = mean(Rq_per_bin) 
    
    return Ra_per_bin, Rq_per_bin, Ra, Rq


#=============================================================================
def acousticDiversityIndex (Sxx, fn, fmin=0, fmax=20000, bin_step=1000, 
                            dB_threshold=3, index="shannon", R_compatible = 'soundecology'):
    
    """
    Acoustic Diversity Index : ADI
    
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
    
    dB_threshold : scalar, optional, default is 3dB
        Threshold to compute the score (ie. the number of data > threshold,
        normalized by the length)
        
    index : string, optional, default is "shannon"
        "shannon" : Shannon entropy is calculated on the vector of scores
        "simpson" : Simpson index is calculated on the vector of scores
        "invsimpson" : Inverse Simpson index is calculated on the vector 
                        of scores
        
    Returns
    -------    
    ADI : scalar 
        Acoustic Diversity Index of the spectrogram (ie. index of the vector 
        of scores)
    
    Reference
    --------
    Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. 
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    """
        
    # number of frequency intervals to compute the score
    N = np.floor((fmax-fmin)/bin_step)
    
    if R_compatible == 'soundecology' :
        # convert into dB and normalization by the max
        Sxx_dB = linear2dB(Sxx/max(Sxx), mode='amplitude')
    else :
        # convert into dB 
        Sxx_dB = linear2dB(Sxx, mode='amplitude')         
    
    # Score for each frequency in the frequency bandwith
    s_sum = []
    for ii in np.arange(0,N):
        f0 = int(fmin+bin_step*(ii))
        f1 = int(f0+bin_step)
        s,_ = score(Sxx_dB[index_bw(fn,(f0,f1)),:], threshold=dB_threshold, axis=0)
        s_sum.append(mean(s))
    
    s = np.asarray(s_sum)
    
    # Entropy
    if index =="shannon":
        ADI = shannonEntropy(s)
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
def acousticEvenessIndex (Sxx, fn, fmin=0, fmax=20000, bin_step=500, 
                          dB_threshold=-50, R_compatible = 'soundecology'):
    
    """
    Acoustic Eveness Index : AEI
    
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
        
    Reference 
    ---------
    Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. 
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    """

    # number of frequency intervals to compute the score
    N = np.floor((fmax-fmin)/bin_step)
    
    if R_compatible == 'soundecology' :
        # convert into dB and normalization by the max
        Sxx_dB = linear2dB(Sxx/max(Sxx), mode='amplitude')
    else :
        # convert into dB 
        Sxx_dB = linear2dB(Sxx, mode='amplitude')   
    
    # Score for each frequency in the frequency bandwith
    s_sum = []
    for ii in np.arange(0,N):
        f0 = int(fmin+bin_step*(ii))
        f1 = int(f0+bin_step)
        s,_ = score(Sxx_dB[index_bw(fn,(f0,f1)),:], threshold=dB_threshold, axis=0)
        s_sum.append(mean(s))
    
    s = np.asarray(s_sum)
    
    # Gini
    AEI = gini(s)
    
    return AEI

#=============================================================================
"""
    Indices based on the entropy
"""
def spectral_entropy (X, fn, frange=None, display=False) :
    """
    Spectral entropy : EAS, ECU, ECV, EPS, 
    
    + kurtosis and skewness : KURT, SKEW
    
    Parameters
    ----------
    X : ndarray of floats
        Spectrum (1d) or Spectrogram (2d). 
        Better to use the PSD to be consistent with energy
    
    fn : 1d ndarray of floats
        frequency vector
    
    frange : tupple (fmin, fmax), optional, default is None
        Frequency band used to compute the spectral entropy.
        For instance, one may want to compute the spectral entropy for the 
        biophony bandwidth
    
    display : boolean, optional, default is False
        Display the different spectra (mean, variance, covariance, max...)
        
    Returns
    -------    
    EAS, ECU, ECV, EPS, KURT, SKEW : scalars 
        EAS : Entropy of spectrum
        ECU : Entropy of spectral variance (along the time axis for each frequency)
        ECV : Entropy of coefficient of variance (along the time axis for each frequency)
        EPS : Entropy of spectral maxima 
        KURT : Kurtosis of spectral maxima
        SKEW : Skewness of spectral maxima
        
    Reference 
    ---------
    Credit : 
    
    """
    
    if isinstance(frange, numbers.Number) :
        print ("WARNING: frange must be a tupple (fmin, fmax) or None")
        return
    
    if frange is None : frange=(fn.min(),fn.max())
    
    # select the indices corresponding to the frequency range
    iBAND = index_bw(fn, frange)

    # TOWSEY & BUXTON : only on the bio band
    """ EAS [TOWSEY] """
    """ 
        COMMENT : Result a bit different due to different Hilbert implementation
    """
    X_mean = mean(X[iBAND], axis=1)
    Hf = entropy(X_mean)
    EAS = 1 - Hf

    #### Entropy of spectral variance (along the time axis for each frequency)
    """ ECU [TOWSEY] """
    X_Var = np.var(X[iBAND], axis=1)
    Hf_var = entropy(X_Var)
    ECU = 1 - Hf_var

    #### Entropy of coefficient of variance (along the time axis for each frequency)
    """ ECV [TOWSEY] """
    X_CoV = np.var(X[iBAND], axis=1)/max(X[iBAND], axis=1)
    Hf_CoV = entropy(X_CoV)
    ECV = 1 - Hf_CoV
    
    #### Entropy of spectral maxima 
    """ EPS [TOWSEY]  """
    ioffset = np.argmax(iBAND==True)
    Nbins = sum(iBAND==True)    
    imax_X = np.argmax(X[iBAND],axis=0) + ioffset
    imax_X = fn[imax_X]
    max_X_bin, bin_edges = np.histogram(imax_X, bins=Nbins, range=frange)
    max_X_bin = max_X_bin/sum(max_X_bin)
    Hf_fmax = entropy(max_X_bin)
    EPS = 1 - Hf_fmax    
    
    #### Kurtosis of spectral maxima
    KURT = kurtosis(max_X_bin)
    
    #### skewness of spectral maxima
    SKEW = skewness(max_X_bin)
    
    if display: 
        fig, ax = plt.subplots()
        ax.plot(fn[iBAND], X_mean/max(X_mean),label="Normalized mean Axx")
        plt.plot(fn[iBAND], X_Var/max(X_Var),label="Normalized variance Axx")
        ax.plot(fn[iBAND], X_CoV/max(X_CoV),label="Normalized covariance Axx")
        ax.plot(fn[iBAND], max_X_bin/max(max_X_bin),label="Normalized Spectral max Axx")
        ax.set_title('Signals')
        ax.set_xlabel('Frequency [Hz]')
        ax.legend()

    return EAS, ECU, ECV, EPS, KURT, SKEW


#=============================================================================
"""
    Indices based on the energy
"""
    
def _energy_per_freqbin (PSDxx, fn, frange = (0, 20000), bin_step = 1000):
        
    #Convert into bins
    PSDxx_bins, bins = intoBins(PSDxx, fn, bin_min=0, bin_max=fn[-1], 
                              bin_step=bin_step, axis=0)   
    
    # select the indices corresponding to the frequency bins range
    indf = index_bw (bins, frange) 

    # select the frequency bins and take the min
    energy = sum(PSDxx_bins[indf, ])
    
    return energy

#=============================================================================
def soundscapeIndex (Sxx,fn,frange_bioPh=(1000,10000),frange_antroPh=(0,1000), 
                     step=None):
    """
    soundscapeIndex
        
    Parameters
    ----------
    Sxx : ndarray of floats
        2d : Amplitude Spectrogram
    
    fn : vector
        frequency vector 
        
    frange_bioPh : tupple (fmin, fmax), optional, default is (1000,10000)
        Frequency band of the biophony
    
    frange_antroPh: tupple (fmin, fmax), optional, default is (0,1000)
        Frequency band of the anthropophony
    
    step: optional, default is None
        if step is None, keep the original frequency resolution, otherwise,
        the spectrogram is converted into new frequency bins
        
    Returns :
    ---------
    NDSI : scalar
        (bioPh-antroPh)/(bioPh+antroPh)
    ratioBA : scalar
        biophonic energy / anthropophonic energy
    antroPh : scalar
        Acoustic energy in the anthropophonic bandwidth
    bioPh : scalar
        Acoustic energy in the biophonic bandwidth
    
    Reference :
    -----------
    Kasten, Eric P., Stuart H. Gage, Jordan Fox, and Wooyeong Joo. 2012. 
    The Remote Environmental Assessment Laboratory's Acoustic Library: An Archive 
    for Studying Soundscape Ecology. Ecological Informatics 12: 50-67.
    
    Inspired by Seewave R package
    """

    # Frequency resolution
    # if step is None, keep the original frequency resolution, otherwise,
    # the spectrogram is converted into new frequency bins
    if step is None : step = fn[1]-fn[0]
    
    # Convert Sxx (amplitude) into PSDxx (energy)
    PSDxx = Sxx**2

    # Energy in BIOBAND
    bioPh = _energy_per_freqbin(PSDxx, fn, frange=frange_bioPh, bin_step=step)
    # Energy in ANTHROPOBAND
    antroPh = _energy_per_freqbin(PSDxx, fn, frange=frange_antroPh, bin_step=step)
    
    # NDSI and ratioBA indices 
    NDSI = (bioPh-antroPh)/(bioPh+antroPh)
    ratioBA = bioPh / antroPh
    
    return NDSI, ratioBA, antroPh, bioPh

#=============================================================================
def bioacousticsIndex (Sxx, fn, frange=(2000, 15000), R_compatible = 'soundecology'):
    """
    Bioacoustics Index
    
    Parameters
    ----------
    Sxx : ndarray of floats
        matrix : Spectrogram  
    
    fn : vector
        frequency vector 
    
    frange : tupple (fmin, fmax), optional, default is (2000, 15000)
        Frequency band used to compute the bioacoustic index.
        
    R_compatible : Boolean, optional, default is False
        if True, the result is similar to the package SoundEcology in R but 
    
    Returns:
    --------
    BI : scalar
        Bioacoustics Index
    
    Reference 
    ---------
    Reference: Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic 
    invasion resistance in Hawaii: bioacoustics, field surveys, and airborne 
    remote sensing. Ecological Applications 17: 2137-2144.
    
    Ported and modified from the soundecology R package.
    
    Erratum
    -------
    Soundecology compatible version
        - average of dB value
        - remove negative value in order to get positive values only
        - dividing by the frequency resolution df instead of multiplication
        ...
    """    
    
    # select the indices corresponding to the frequency bins range
    indf = index_bw(fn,frange)
    
    # frequency resolution. 
    df = fn[1] - fn[0]
    
    # ======= As soundecology
    if R_compatible == 'soundecology' :
        # Mean Sxx normalized by the max
        meanSxx = mean(Sxx/max(Sxx), axis=1)
        # Convert into dB
        meanSxxdB = linear2dB(meanSxx, mode='amplitude')
        
        # "normalization" in order to get positive 'vectical' values 
        meanSxxdB = meanSxxdB[indf,]-min(meanSxxdB[indf,])
    
        # this is not the area under the curve...
        # what is the meaning of an area under the curve in dB...
        BI = sum(meanSxxdB)/df
        
    else:
        # normalize by the max of the spectrogram
        # better to average the PSD for energy conservation
        PSDxx_norm = (Sxx/max(Sxx))**2
        meanPSDxx_norm = mean(PSDxx_norm, axis=1)

        # Compute the area
        # take the sqrt in order to go back to Sxx
        BI = sqrt(sum(meanPSDxx_norm))* df 
        
    return BI
    
"""
    Indices based on the acoustic event
    
"""

def acoustic_activity (xdB, dB_threshold, axis=1):
    """
    Acoustic Activity :
        for each frequency bin :
            - ACTfract : proportion (fraction) of points above the threshold 
            - ACTcount : number of points above the threshold
            - ACTmean : mean value (in dB) of the portion of the signal above the threhold
    Parameters
    ----------
    xdB : ndarray of floats
        1d : envelope in dB
        2d : PSD spectrogram in dB
        It's better to work with PSD or envelope without background variation
        as the process is based on threshold.

    dt : scalar
        Time resolution

    dB_threshold : scalar, optional, default is 6dB
        data >Threshold is considered to be an event 
        if the length is > rejectLength
        
    rejectDuration : scalar, optional, default is None
        event shorter than rejectDuration are discarded
    
    Returns
    -------    
    ACTfract :ndarray of scalars
        proportion (fraction) of points above the threshold for each frequency bin
    ACTcount: ndarray of scalars
        number of points above the threshold for each frequency bin
    ACTmean: scalar
        mean value (in dB) of the portion of the signal above the threhold
        
    Reference 
    ---------
    Towsey, Michael W. (2013) Noise removal from wave-forms and spectrograms derived 
    from natural recordings of the environment.
    Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived 
    from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    
        ACTsp [Towsey] : ACTfract (proportion (fraction) of point value above the theshold)
        EVNsp [Towsey] : ACTcount (number of point value above the theshold)
    """ 
    ACTfract, ACTcount = score(xdB, dB_threshold, axis=axis)
    ACTfract= ACTfract.tolist()
    ACTcount = ACTcount.tolist()
    ACTmean = dB2linear(xdB[xdB>dB_threshold], mode='power')
    return ACTfract, ACTcount, ACTmean
       

#=============================================================================     
def acoustic_events(xdB, dt, dB_threshold=6, rejectDuration=None):
    """
    Acoustic events :
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
    
    Returns
    -------    
    EVNsum :scalar
    EVNmean: scalar
    EVNcount: scalar
    EVN: ndarray of floats 

    Reference 
    ---------
    Towsey, Michael W. (2013) Noise removal from wave-forms and spectrograms derived 
    from natural recordings of the environment.
    Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived 
    from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
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
