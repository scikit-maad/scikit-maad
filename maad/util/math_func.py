#!/usr/bin/env python
""" 
Mathematical tools for audio signal processing.
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
import matplotlib.pyplot as plt
import numpy as np 
from numpy import mean, median, var
from scipy.ndimage.filters import uniform_filter1d # for fast running mean
from scipy.signal import periodogram, welch
import pandas as pd
# min value
import sys
_MIN_ = sys.float_info.min

# Import internal modules
from maad.util import linear_scale


#%%
# =============================================================================
# public functions
# =============================================================================

def running_mean(x, N, mode="nearest"):
    """
    Compute fast running mean for a window size N.
    
    Parameters
    ----------
    x :  1d ndarray of scalars
        Vector 

    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional, 
        The `mode` parameter determines how the input array is extended
        when the filter overlaps a border. Default is 'nearest'. Behavior
        for each valid value is as follows:
    
        'reflect' (`d c b a | a b c d | d c b a`)
            The input is extended by reflecting about the edge of the last
            pixel.
    
        'constant' (`k k k k | a b c d | k k k k`)
            The input is extended by filling all values beyond the edge with
            the same constant value, defined by the `cval` parameter.
    
        'nearest' (`a a a a | a b c d | d d d d`)
            The input is extended by replicating the last pixel.
    
        'mirror' (`d c b | a b c d | c b a`)
            The input is extended by reflecting about the center of the last
            pixel.
    
        'wrap' (`a b c d | a b c d | a b c d`)
            The input is extended by wrapping around to the opposite edge.
            
    N : int
        length of window to compute the mean
        
    Returns
    -------
    x_mean : 1d ndarray of scalars
        Vector with the same dimensions than the original variable x
        
    Examples
    --------
    >>> maad.util.running_mean([2, 8, 0, 4, 1, 9, 9, 0], N=3)
        array([4, 3, 4, 1, 4, 6, 6, 3])
        
    """         
    x_mean = uniform_filter1d(x, size=N, mode="nearest")
    return x_mean

#%%
def get_unimode (X, mode ='median', axis=1, N=7, N_bins=100, verbose=False):
    """
    Get the statistical mode or modal value which is the most common number in the 
    dataset.
    
    Parameters
    ----------
    X :  1d or 2d ndarray of scalars
        Vector or matrix 
                
    mode : str, optional, default is 'median'
        Select the mode to remove the noise
        Possible values for mode are :
        - 'ale' : Adaptative Level Equalization algorithm [Lamel & al. 1981]
        - 'median' : subtract the median value
        - 'mean' : subtract the mean value (DC)
    
    axis : integer, default is 1
        if matrix, estimate the mode for each row (axis=0) or each column (axis=1)
        
    N : int (only for mode = "ale")
        length of window to compute the running mean of the histogram
        
    N_bins : int (only for mode = "ale")
        number of bins to compute the histogram
            
    verbose : boolean, optional, default is False
        print messages into the consol or terminal if verbose is True
                      
    Returns
    -------
    unimode_value : float
        The most common number in the dataset
        
    Notes
    -----
    ale : Adaptative Level Equalization algorithm from Lamel et al., 1981 :
    L.F. Lamel, L.R. Rabiner, A.E. Rosenberg, J.G. Wilpon
    An improved endpoint detector for isolated word recognition
    IEEE Trans. ASSP, ASSP-29 (1981), pp. 777-785
    
    Examples
    --------
    
    This function is interesting to obtain the background noise (BGN) profile (e.g. frequency bin
    by frequency bin) of a spectrogram
    
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(w,fs,window='hanning',noverlap=512, nFFT=1024)
    >>> Sxx_dB = maad.util.power2dB(Sxx_power)
    >>> BGN_med = maad.util.get_unimode (Sxx_dB, mode='median', axis=1)
    >>> import matplotlib.pyplot as plt 
    >>> plt.plot(fn,maad.util.mean_dB(Sxx_dB,axis=1))
    >>> plt.plot(fn,BGN_med)
    
    Extract the background noise from mean
    
    >>> BGN_mean = maad.util.get_unimode (Sxx_dB, mode='mean', axis=1)
    >>> plt.plot(fn,BGN_mean)
    
    Extract the background noise from ale (i.e. unimode)
    
    >>> BGN_ale = maad.util.get_unimode (Sxx_dB, mode='ale', N=7, N_bins=100, axis=1)
    >>> plt.plot(fn,BGN_ale)
    
    """         
    if X.ndim ==2: 
        if axis == 0:
            X = X.transpose()
            axis = 1
    elif X.ndim ==1: 
        axis = 0
        
    if mode=='ale':
                
        if X.ndim ==2:
            unimode_value = []
            for i, x in enumerate(X):  
                # Min and Max of the envelope (without taking into account nan)
                x_min = np.nanmin(x)
                x_max = np.nanmax(x)
                # Compute a 50-bin histogram ranging between Min and Max values
                hist, bin_edges = np.histogram(x, bins=N_bins, range=(x_min, x_max))
                
                # smooth the histogram by running mean
                hist_smooth = running_mean(hist, N, mode="nearest")
                    
                # find the maximum of the peak with quadratic interpolation
                # don't take into account the first 4 bins.
                imax = np.argmax(hist_smooth) 
               
                unimode_value.append(bin_edges[imax])
                
            # transpose the vector
            unimode_value = np.asarray(unimode_value)
            unimode_value = unimode_value.transpose()
        else:
            x = X
            # Min and Max of the envelope (without taking into account nan)
            x_min = np.nanmin(x)
            x_max = np.nanmax(x)
            
            # Compute a 50-bin histogram ranging between Min and Max values
            hist, bin_edges = np.histogram(x, bins=N_bins, range=(x_min, x_max))
            
            # smooth the histogram by running mean
            hist_smooth = running_mean(hist, N, mode="nearest")
                                                 
            # find the maximum of the peak with quadratic interpolation
            imax = np.argmax(hist_smooth)
                        
            # assuming an additive noise model : noise_bckg is the max of the histogram
            # as it is an histogram, the value is unimode_value = bin_edges_interp[np.argmax(hist_interp)]
            unimode_value = bin_edges[imax]

    elif mode=='median':
        unimode_value = median(X, axis=axis)
        
    elif mode=='mean':
        unimode_value = mean(X, axis=axis)
        
    return unimode_value

#%%
def rms(s):
    """
    Compute the root-mean-square (RMS) level of an input signal. 
    
    RMS is defined as the square root of the arithmetic mean of the square of a set of numbers [1]. The RMS is used to estimate de mean amplitude level of an audio signal or any alternative time series.
    
    
    Parameters
    ----------
    s : 1D array
        Input signal to process

    Returns
    -------
    rms: float
        Root mean square of input signal
    
    References
    ----------
    .. [1] 'Root mean square' (2010). Wikipedia. Available at https://en.wikipedia.org/wiki/Root_mean_square
    
    Examples
    --------
    >>> from maad import sound, util
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> rms_value = util.rms(s)
    
    """
    return np.sqrt(np.mean(s**2))

#%%
def skewness (x, axis=None):
    """
    Compute the skewness (asymetry) of an audio signal.
    
    Parameters
    ----------
    x : ndarray of floats 
        1d signal or 2d matrix
    axis : integer, optional, default is None
        select the axis to compute the kurtosis
        The default is to compute the mean of the flattened array.
                            
    Returns
    -------    
    sk : float or ndarray of floats
        skewness of x 
        if x is a 1d vector => single value
        if x is a 2d matrix => array of values corresponding to the number of
        points in the other axis
    
    Examples
    --------
    >>> from maad import sound, util
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> util.skewness(s)
    -0.006547980427883208
    
    """
    if isinstance(x, (np.ndarray)) == True:
        if axis is None:
            # flatten the array
            Nf = len(np.ndarray.flatten((x)))
        else:
            Nf = x.shape[axis]
        mean_x =  np.mean(x, axis=axis)
        std_x = np.std(x, axis=axis)
        if axis == 0 :
            z = x - mean_x[np.newaxis, ...]
        else :
            z = x - mean_x[..., np.newaxis]
        sk = (np.sum(z**3, axis=axis)/(Nf-1))/std_x**3
    else:
        print ("WARNING: type of x must be ndarray") 
        sk = None

    # test if ku is an array with a single value
    if (isinstance(sk, (np.ndarray)) == True) and (len(sk) == 1):
        sk = float(sk)

    return sk

#%%
def kurtosis (x, axis=None):
    """
    Compute the kurtosis (tailedness or curved or arching) of an audio signal.
    
    Parameters
    ----------
    x : ndarray of floats 
        1d signal or 2d matrix       
    axis : integer, optional, default is None
        select the axis to compute the kurtosis
        The default is to compute the mean of the flattened array.
                            
    Returns
    -------    
    ku : float or ndarray of floats
        kurtosis of x 
        if x is a 1d vector => single value
        if x is a 2d matrix => array of values corresponding to the number of
        points in the other axis
        
    Examples
    --------
    >>> from maad import sound, util
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> util.kurtosis(s)
    24.711610834321217        
        
    """
    if isinstance(x, (np.ndarray)) == True:
        if axis is None:
            # flatten the array
            Nf = len(np.ndarray.flatten((x)))
        else:
            Nf = x.shape[axis]
        mean_x =  np.mean(x, axis=axis)
        std_x = np.std(x, axis=axis)
        if axis==0 :
            z = x - mean_x[np.newaxis, ...]
        else:
            z = x - mean_x[..., np.newaxis]
        ku = (np.sum(z**4, axis=axis)/(Nf-1))/std_x**4
    else:
        print ("WARNING: type of x must be ndarray") 
        ku = None
        
    # test if ku is an array with a single value
    if (isinstance(ku, (np.ndarray)) == True) and (len(ku) == 1):
        ku = float(ku)
       
    return ku

#%%
def moments (X, axis=None):
    """
    Computes the first 4th moments of a vector (1d, ie. spectrum or waveform) 
    or spectrogram (2d) 
    
    - mean
    - variance
    - skewness
    - kurtosis
    
    Parameters
    ----------
    X : ndarray of floats
        vector (1d : spectrum, waveform) or matrix (2d : spectrogram). 
    axis : interger, optional, default is None
        if spectrogram (2d), select the axis to estimate the moments.
        
    Returns
    -------
    mean : float 
        mean of X
    var : float 
        variance  of X
    skew : float
        skewness of X
    kurt : float
        kurtosis of X

    Examples
    --------
    >>> from maad import sound, util
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> mean, var, skew, kurt = util.moments(s)
    >>> print ('mean:%2.4f / var:%2.4f / skew:%2.4f / kurt:%2.4f' %(mean, var, skew, kurt)) 
    mean:-0.0000 / var:0.0012 / skew:-0.0065 / kurt:24.7116    
    """
    # force P to be ndarray
    X = np.asarray(X)
    
    return mean(X, axis), var(X, axis), skewness(X, axis), kurtosis(X, axis)

#%%
def entropy (x, axis=0):
    """
    Compute the entropy of a vector (waveform) or matrix (spectrogram).
    
    Parameters
    ----------
    x : ndarray of floats
        x is a vector (1d) or a matrix (2d)

    axis : int, optional, default is 0
        select the axis where the entropy is computed
        if x is a vector, axis=0
        if x is a 2d ndarray, axis=0 => rows, axis=1 => columns
                
    Returns
    -------
    H : float or ndarray of floats
        entropy of x
        
    Examples
    --------
    >>> from maad import sound, util
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> H = util.entropy(s)
    >>> print ('Entropy is %2.4f' %H) 
    Entropy is 0.9998 
        
    """
    if isinstance(x, (np.ndarray)) == True:
        if x.ndim > axis:
            if x.shape[axis] == 0: 
                print ("WARNING: x is empty") 
                H = None 
            elif x.shape[axis] == 1:
                H = 0 # null entropy
            elif x.all() == 0:
                if x.ndim == 1 : # case vector
                    H = 0 # null entropy
                else : # case matrix
                    if axis == 0 : H = np.zeros(x.shape[1]) # null entropy
                    if axis == 1 : H = np.zeros(x.shape[0]) # null entropy
            else:
                # if datain contains negative values -> rescale the signal between 
                # between posSitive values (for example (0,1))
                if np.min(x)<0:
                    x = linear_scale(x,minval=0,maxval=1)
                # length of datain along axis
                n = x.shape[axis]
                # Tranform the signal into a Probability mass function (pmf)
                # Sum(pmf) = 1
                if axis == 0 :
                    pmf = x/np.sum(x,axis)
                elif axis == 1 :                     
                    pmf = (x.transpose()/np.sum(x,axis)).transpose()
                pmf[pmf==0] = _MIN_
                #normalized by the length : H=>[0,1]
                H = -np.sum(pmf*np.log(pmf),axis)/np.log(n)
        else:
            print ("WARNING :axis is greater than the dimension of the array")    
            H = None 
    else:
        print ("WARNING: x must be ndarray")   
        H = None 

    return H
