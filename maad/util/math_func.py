#!/usr/bin/env python
""" Utilitary functions for scikit-MAAD """
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
from numpy import mean, median
from scipy.ndimage.filters import uniform_filter1d # for fast running mean
import matplotlib.pyplot as plt

# min value
import sys
_MIN_ = sys.float_info.min


#=============================================================================
def running_mean(x, N, mode="nearest"):
    """
    Compute fast running mean for a window size N
    
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
        Vector
        
    Examples
    --------
    >>> maad.util.running_mean([2, 8, 0, 4, 1, 9, 9, 0], N=3)
        array([4, 3, 4, 1, 4, 6, 6, 3])
        
    """         
    x_mean = uniform_filter1d(x, size=N, mode="nearest")
    return x_mean

#=============================================================================
def get_unimode (X, mode ='mean', axis=1, N=7, N_bins=100, verbose=False):
    """
    determine the statistical mode or modal value which is 
    the most common number in the dataset
    
    Parameters
    ----------
    X :  1d or 2d ndarray of scalar
        Vector or matrix 
                
    mode : str, optional, default is 'mean'
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
    -------
    
    This function is interesting to obtain the background noise (BGN) profile (e.g. frequency bin
    by frequency bin) of a spectrogram
    
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> PSDxx,tn,fn,_ = maad.sound.spectrogram(w,fs,window='hanning',noverlap=512, nFFT=1024)
    >>> PSDxx_dB = maad.util.energy2dBSPL(PSDxx)
    >>> BGN_med = maad.util.get_unimode (PSDxx_dB, mode ='median', axis=1, 
                                               verbose=False, display=False)
    >>> import matplotlib.pyplot as plt 
    >>> plt.plot(fn,maad.util.energy2dBSPL(mean(PSDxx,axis=1)))
    >>> plt.plot(fn,BGN_med)
    
    Extract the background noise from mean
    
    >>> BGN_mean = maad.util.get_unimode (PSDxx_dB, mode ='mean', axis=1, 
                                               verbose=False, display=False)
    >>> plt.plot(fn,BGN_mean)
    
    Extract the background noise from ale (i.e. unimode)
    
    >>> BGN_ale = maad.util.get_unimode (PSDxx_dB, mode ='ale', N=7, N_bins=100, axis=1, 
                                               verbose=False, display=False)
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



