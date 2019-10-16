#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Created on Wed Oct 24 11:56:28 2018
    Alpha indices used in ecoacoustics"""
#
# Authors:  Sylvain HAUPERT <sylvain.haupert@mnhn.fr>             
#
# License: New BSD License
    
# Clear all the variables 
#from IPython import get_ipython
#get_ipython().magic('reset -sf')

"""****************************************************************************
# -------------------       Load modules            ---------------------------
****************************************************************************"""

# Import external modules
import numbers
import math

import numpy as np 
from numpy import sum, log, log10, min, max, abs, mean, median, sqrt, diff

import scipy as sp
from scipy.signal import hilbert, hann, sosfilt, convolve, iirfilter, get_window, resample, tukey
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.interpolate import interp1d 
from scipy.stats import rankdata
from scipy import ndimage as ndi


from skimage import filters,transform, measure

import matplotlib.pyplot as plt
# min value
#_MIN_ = sys.float_info.min
_MIN_ = 1e-10

TEST=False

# =============================================================================
# List of functions
# =============================================================================

def shift_bit_length(x):
    """
         find the closest power of 2 that is superior or equal to the number x
    """
    return 1<<(x-1).bit_length()

#=============================================================================
       
def rle(x):
    """
        Run--Length encoding    
        from rle function R
    """
    x = np.asarray(x)
    if x.ndim >1 : 
        print("x must be a vector")
    else:
        n = len(x)
        states = x[1:] != x[:-1]
        i = np.r_[np.where(states)[0], n-1]
        lengths = np.r_[i[0], diff(i)]
        values = x[i]
    return lengths, values
  
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
def spectrogram (wave, fs, N=512, noverlap=0, mode='amplitude', detrend=False, 
                 verbose=False, display=False):
    """
    Convert a sound waveform into a serie of frequency frames (ie matrix) of
    length N/2 (ie number of frequency bins)
    => this is equivalent to calculate a spectrogram of a waveform
    
    Parameters
    ----------
    wave : 1d ndarray of floats (already divided by the number of bits)
        Vector containing the sound waveform 
        
    fs : int
        The sampling frequency in Hz       
        
    N : int, optional, default: 512
        This number is used to compute the short fourier transform (sfft).
        For fast calculation, it's better to use a number that is a power 2. 
        This parameter sets the resolution in frequency as the spectrogram will
        contains N/2 frequency bins between 0Hz-(fs/2)Hz, with a resolution
        df = fs/N
        It sets also the time slot (dt) of each frequency frames : dt = N/fs
        The higher is the number, the lower is the resolution in time (dt) 
        but better is the resolution in frequency (df).
    
    method : str, optional, default is 'stft'
        select the method of calculation of the spectrogram
            - 'stft' : short Fourier Transform
            - 'welsh': welsh method
            => the time dimension is different at the end
            
    mode : str, optional, default is 'amplitude'
        select the output values of spectrogram :
            - 'amplitude' : Sxx = A
            - 'energy'    : Sxx = A² (= Power Spectrum Density (PSD))
          
    Returns
    -------
    Sxx : 2d ndarray of floats
        Spectrogram : Matrix containing K frames with N/2 frequency bins, 
        K*N <= length (wave)
    """
    
    # transform wave into array
    wave = np.asarray(wave)
    
    # compute the number of frames
    K = len(wave)//N
    
    nperseg = N
    
    # sliding window 
    #win = tukey(nperseg, 1/32)
    #win = np.ones(nperseg)
    win=hann(nperseg)
    
    if verbose:
        print(72 * '_')
        print("Computing spectrogram with nperseg=%d and noverlap=%d..." 
              % (nperseg, noverlap))
    
    if mode == 'amplitude':
        fn, tn, Sxx = sp.signal.spectrogram(wave, fs, win, nperseg=nperseg, 
                                            noverlap=noverlap, nfft=nperseg, 
                                            scaling ='spectrum', mode='complex', 
                                            detrend=detrend)    
    if mode == 'psd':
        fn, tn, Sxx = sp.signal.spectrogram(wave, fs, win, nperseg=nperseg, 
                                            noverlap=noverlap, nfft=nperseg, 
                                            scaling ='spectrum', mode='psd', 
                                            detrend=detrend)          
        
    # Get the magnitude of the complex 
    Sxx = abs(Sxx)  

    if mode == 'amplitude':
        scale_stft = np.mean(win) 
        Sxx = Sxx / scale_stft # normalization   
    if mode == 'psd': 
        scale_stft = np.sqrt(1/np.sqrt(np.mean(win**2)))
        Sxx = Sxx / scale_stft  # normalization        
   
    # test if the last frames are computed on a whole time frame. 
    # if note => remove these frames
    if Sxx.shape[1] > K:
        sup = Sxx.shape[1] - K
        Sxx = Sxx[:,:-sup]
        tn = tn[:-sup]
        
    # Remove the last frequency bin in order to obtain nperseg/2 frequency bins
    # instead of nperseg/2 + 1 
    Sxx = Sxx[:-1,:]
    fn = fn[:-1]

    if verbose:
        print('max value of the spectrogram %.5f' % Sxx.max())

    # dt and df resolution
    dt = tn[1]-tn[0]
    df = fn[1]-fn[0]
    if verbose:
        print("*************************************************************")
        print("   Time resolution dt=%.2fs | Frequency resolution df=%.2fHz "
              % (dt, df))  
        print("*************************************************************")
        
    # display full SPECTROGRAM in dB
    if display :
        
        # transform data in dB
        if mode == 'psd':
            Sxx = sqrt(Sxx)
        #### convert into dB
        SxxdB = linear2dB(Sxx, db_range=MIN_dB, db_gain=0)
        
        fig, ax = plt.subplots()
        # set the paramteers of the figure
        fig.set_facecolor('w')
        fig.set_edgecolor('k')
        fig.set_figheight(4)
        fig.set_figwidth (13)
                
        # display image
        _im = ax.imshow(SxxdB, extent=(tn[0], tn[-1], fn[0], fn[-1]), 
                         interpolation='none', origin='lower', 
                         vmin =-MIN_dB, vmax=0, cmap='gray')
        plt.colorbar(_im, ax=ax)
 
        # set the parameters of the subplot
        ax.set_title('Spectrogram')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Frequency [Hz]')
        ax.axis('tight') 
     
        fig.tight_layout()
         
        # Display the figure now
        plt.show()    
        
    return Sxx, tn, fn


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
    if bin_step <= (an[1]-an[0]):
        raise Exception('WARNING: bin step must be larger than the actual resolution of x')

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
def linearScale(x, minval= 0.0, maxval=1.0):
    """ 
    Program to scale the values of a matrix from a user specified minimum to a user specified maximum
    
    Parameters
    ----------
    x : array-like
        numpy.array like with numbers
    minval : scalar, optional, default : 0
        This minimum value is attributed to the minimum value of the array 
    maxval : scalar, optional, default : 1
        This maximum value is attributed to the maximum value of the array         
        
    Returns
    -------
    y : array-like
        numpy.array like with numbers  
        
    -------
    Example
    -------
        a = np.array([1,2,3,4,5]);
        a_out = linearScale(a,0,1);
    Out: 
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        
    References:
    ----------
    Program written by Aniruddha Kembhavi, July 11, 2007 for MATLAB
    Adapted by S. Haupert Dec 12, 2017 for Python
    """
    # if x is a list, convert x into ndarray 
    if isinstance(x, list):
        x = np.asarray(x)
        
    y = x - x.min();
    y = (y/y.max())*(maxval-minval);
    y = y + minval;
    return y

#=============================================================================
def linear2dB (x, db_range=None, db_gain=0):
    """
    Transform linear date into decibel scale within the dB range (db_range).
    A gain (db_gain) could be added at the end.    
    
    Parameters
    ----------
    x : array-like
        data to rescale in dB  
    db_range : scalar, optional, default : None
        if db_range is a number, anything lower than -db_range is set to 
        -db_range and anything larger than 0 is set to 0
    db_gain : scalar, optional, default is 0
        Gain added to the results 
        amplitude --> 20*log10(x) + db_gain  
    Returns
    -------
    y : scalars
        amplitude--> 20*log10(x) + db_gain  

    """            
    x = abs(x)   # take the absolute value of datain
    
    # Avoid zero value for log10  
    # if it's a scalar
    if hasattr(x, "__len__") == False:
        if x ==0: x = _MIN_  
    else :     
        x[x ==0] = _MIN_  # Avoid zero value for log10 

    # conversion in dB     
    y = 20*log10(x)   # take log
    
    if db_gain : y = y + db_gain    # Add gain if needed
    
    if db_range is not None :
        # set anything above db_range as 0
        y[y > 0] = 0  
        # set anything less than -db_range as -db_range
        y[y < -(db_range)] = -db_range  
        
    return y

#=============================================================================
def dB2linear (x, db_gain=0):
    """
    Transform linear date into decibel scale within the dB range (db_range).
    A gain (db_gain) could be added at the end.    
    
    Parameters
    ----------
    x : array-like
        data in dB to rescale in linear 

    db_gain : scalar, optional, default is 0
        Gain added to the results 
                --> 20*log10(x) + db_gain
                
    Returns
    -------
    y : scalars
        --> 10^(x/20 - db_gain) 
    """  
    y = 10**((x- db_gain)/20) 
    return y

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
    Translation from R to Python of the function from SEEWAVE [Jérôme Sueur]       
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
                    datain = linearScale(datain,minval=0,maxval=1)
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
    count : the number of times values in x are greater than the threshold.
    s : count is normalized by the total number of values in x
    
    """
    x = np.asarray(x)
    x = x>=threshold
    count = sum(x,axis=axis)
    s = sum(x,axis=axis)/x.shape[axis]
    return s, count

#=============================================================================
def gini(x, corr=False):
    """
    from ineq library in R
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
def iir_filter1d(x, fs, fcut, forder, fname ='butter', ftype='bandpass', rp=None, 
              rs=None):
    """
    x : array_like
        1d vector of scalar to be filtered
        
    fs : scalar
        sampling frequency   
        
    fcut : array_like
        A scalar or length-2 sequence giving the critical frequencies.             
    
    forder : int
        The order of the filter.
    
    ftype : {‘bandpass’, ‘lowpass’, ‘highpass’, ‘bandstop’}, optional, default 
        is ‘bandpass’
        The type of filter.
        
    fname : {‘butter’, ‘cheby1’, ‘cheby2’, ‘ellip’, ‘bessel’}, optional, default
        is ‘butter’
        
    The type of IIR filter to design:
            Butterworth : ‘butter’
            Chebyshev I : ‘cheby1’
            Chebyshev II : ‘cheby2’
            Cauer/elliptic: ‘ellip’
            Bessel/Thomson: ‘bessel’
            
    rp : float, optional
        For Chebyshev and elliptic filters, provides the maximum ripple in 
        the passband. (dB)
        
    rs : float, optional
        For Chebyshev and elliptic filters, provides the minimum attenuation 
        in the stop band. (dB)           
            
    """
    sos = iirfilter(N=forder, Wn=np.asarray(fcut)/(fs/2), btype=ftype,ftype=fname, rp=rp, 
                     rs=rs, output='sos')
    y = sosfilt(sos, x)
    return y
 
#=============================================================================
def fir_filter(x, kernel, axis=0):
    """
    x : array_like
        1d vector or 2d matrix of scalars to be filtered
   
    kernel : array_like or tuple
        Pass directly the kernel (1d vector of scalars) 
        Or pass the arguments in a tuple to create a kernel. Arguments are
                window : string, float, or tuple
                    The type of window to create. 
                - boxcar, triang, blackman, hamming, hann, bartlett, 
                  flattop,parzen, bohman, blackmanharris, nuttall, barthann, 
                - (kaiser, beta), 
                - (gaussian, standard deviation), 
                - (general_gaussian, power, width), 
                - (slepian, width), 
                - (dpss, normalized half-bandwidth), 
                - (chebwin, attenuation), 
                - (exponential, decay scale), 
                - (tukey, taper fraction)
                Nx : int
                    The number of samples in the window.
                    
        examples:
            kernel = ('boxcar', 9)
            kernel = (('gaussian', 0.5), 5)
            kernel = [1 3 5 7 5 3 1] 
    axis : int
        Determine along which axis is performed the filtering in case of 2d matrix
    """
    if isinstance(kernel,tuple) :
        if len(kernel) ==1 :
            win = get_window(kernel[0])
        if len(kernel) ==2 :
            win = get_window(kernel[0], kernel[1])
    elif isinstance(kernel,list) or isinstance(kernel,np.ndarray):
        win = kernel

    if x.ndim == 1:
        # Trick to avoid strange values at the beginning and end of the convolve
        # signal. Add mirror values
        x = np.insert(x, 0, np.flipud(x[0:len(win)//2]), axis=0) 
        x = np.insert(x, len(x), np.flipud(x[-(len(win)//2):]), axis=0) 
        # convolve and normalized the result by the sum of the kernel
        y = convolve(x, win, mode='same') / sum(win)
        y = y[len(win)//2:-(len(win)//2)]
    elif x.ndim == 2:
        
        if axis ==1 : x = x.transpose()
        
        # Trick to avoid strange values at the beginning and end of the convolve
        # signal. Add mirror values
        x = np.insert(x, 0, np.flipud(x[0:len(win)//2]), axis=0) 
        x = np.insert(x,  x.shape[0], np.flipud(x[-(len(win)//2):]), axis=0) 
        y = np.zeros(x.shape)
        for i in np.arange (x.shape[1]):
            y[:,i] = convolve(x[:,i], win, mode='same') / sum(win)
        y = y[len(win)//2:-(len(win)//2),:]
        
        if axis ==1 :y = y.transpose()

    return y

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
    Translation from R to Python of the function AR from SEEWAVE [Jérôme Sueur]       
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
    Described in [Sueur & al. 2008]
    
    !!!!! in Seewave, the result is the sum of the ACI per bin.
    
    """   
    if norm == 'per_bin':
        ACI_xx = ((abs(diff(Sxx,1)).transpose())/(sum(Sxx,1)).transpose()).transpose()
    elif norm == 'global':
        ACI_xx = (abs(diff(Sxx,1))/sum(Sxx))

    ACI_per_bin = sum(ACI_xx,axis=1)
    ACI_sum = sum(ACI_per_bin)
    
    return ACI_xx, ACI_per_bin, ACI_sum, 


def roughnessAsACI (Sxx, norm ='global'):
    
    """
    Roughness as ACi (Second derivative instead of first)
    
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
    r_xx : 2d ndarray of scalars
        ROUGHNESS of the spectrogram
    
    r_per_bin : 1d ndarray of scalars
        ROUGHNESS value for each frequency bin
        sum(r_xx,axis=1)
        
    r_sum : scalar
        Sum of ROUGHNESS value per frequency bin (Common definition)
        sum(r_per_bin)
        
    r_mean ; scalar
        average ROUGHNESS value per frequency bin (independant of the number of 
        frequency bin)
        mean(r_per_bin)
    """    
    if norm == 'per_bin':
        r_xx = (((diff(Sxx,2,1)**2).transpose())/(sum(Sxx,1)).transpose()).transpose()
    elif norm == 'global':
        r_xx = ((diff(Sxx,2,1)**2)/sum(Sxx))
    
    r_per_bin = sum(r_xx,axis=1)
    r_sum = sum(r_per_bin)
    
    return r_xx, r_per_bin, r_sum

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
def acousticGradientIndex(Sxx, dt, order=1, norm=None, n_pyr=1, display=False):
    """
    Acoustic Gradient Index : AGI
    
    !!! Must be calculated on raw spectrogram (background noise must remain)
    
    Parameters
    ----------
    Sxx : ndarray of floats
        2d : Spectrogram (i.e matrix of spectrum)
    
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
           
    """     
    
    AGI_xx_pyr = []
    AGI_per_bin_pyr = []
    AGI_mean_pyr = []
    AGI_sum_pyr = []
    dt_pyr = []
    
    for n in np.arange(0,n_pyr):  
        
#        # Show the Leq energy in order to control that the conservation of
#        # energy is preserved
#        PSDxx_mean = mean(Sxx**2,axis=1)
#        leq = 10*log10(sum(PSDxx_mean)/(20e-6)**2)
#        print(leq)
        
        # derivative (order = 1, 2, 3...)
        AGI_xx = abs(diff(Sxx, order, axis=1)) / (dt**order )
        
        #print('PYRAMID: %d / median ATI: %f / size: %s' % (n, median(AGI_xx), AGI_xx.shape))
        
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
        # Sum Global
        AGI_sum = sum(AGI_per_bin)

        # add to the lists
        AGI_xx_pyr.append(AGI_xx)        
        AGI_per_bin_pyr.append(AGI_per_bin)
        AGI_mean_pyr.append(AGI_mean)
        AGI_sum_pyr.append(AGI_sum)
        dt_pyr.append(dt)

        # build next pyramid level (gaussian filter then reduce)
        # Sigma for gaussian filter. Default is 2 * downscale / 6.0 
        # which corresponds to a filter mask twice the size of the scale factor 
        # that covers more than 99% of the gaussian distribution.
        # The total energy is kept
        dt = dt*2 # the resolution decreases by 2 = x2
        PSDxx = Sxx**2
        # blur the image only on axis 1 (time axis)
        PSDxx_blur = ndi.gaussian_filter1d(PSDxx,axis=1, sigma=2*2/6.0)
        dim = tuple([PSDxx_blur.shape[0], math.ceil(PSDxx_blur.shape[1]/2)])
        #  Reduce the size of the image by 2
        PSDxx_reduced = transform.resize(PSDxx_blur,output_shape=dim, mode='reflect', anti_aliasing=True)      

        # display full SPECTROGRAM in dB
        if display==True :
            
            fig4, ax4 = plt.subplots()
            # set the paramteers of the figure
            fig4.set_facecolor('w')
            fig4.set_edgecolor('k')
            fig4.set_figheight(4)
            fig4.set_figwidth (13)
                    
            # display image
            _im = ax4.imshow(10*log10(PSDxx_reduced/(20e-6)**2), extent=(0, 60, 0, 20000), 
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
        
        # back to amplitude
        Sxx = sqrt(PSDxx_reduced)
        
    return AGI_xx_pyr , AGI_per_bin_pyr, AGI_mean_pyr, AGI_sum_pyr, dt_pyr

#=============================================================================
def acousticDiversityIndex (Sxx_dB, fn, fmin=0, fmax=20000, bin_step=1000, 
                            dB_threshold=3, index="shannon"):
    
    """
    Acoustic Diversity Index : ADI
    
    Parameters
    ----------
    Sxx_dB : ndarray of floats
        2d : Spectrogram  in dB
    
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
    """
        
    # number of frequency intervals to compute the score
    N = np.floor((fmax-fmin)/bin_step)
    
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
def acousticEvenessIndex (Sxx_dB, fn, fmin=0, fmax=20000, bin_step=500, 
                          dB_threshold=-50):
    
    """
    Acoustic Eveness Index : AEI
    
    Parameters
    ----------
    Sxx_dB : ndarray of floats
        2d : Spectrogram  in dB
    
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
    """

    # number of frequency intervals to compute the score
    N = np.floor((fmax-fmin)/bin_step)
    
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
def backgroundNoise (X, mode ='ale',axis=1, verbose=False, display=False):
    """
    determine the background noise level in a audiogram or spectrogram
    
    Parameters
    ----------
    X :  1d or 2d ndarray of scalar
        Vector or matrix containing the envelope or the spectrogram
                
    mode : str, optional, default is 'ale'
        Select the mode to remove the noise
        Possible values for mode are :
            - 'ale' : Adaptative Level Equalisation algorithm [Lamel & al. 1981]
                      Background noise value is set equal to the distribution 
                      energy values in the audiogram [Towsey QUT print 2017]
            - 'median' : subtract the median value
            - 'mean' : subtract the mean value (DC)
            
    verbose : boolean, optional, default is False
        print messages into the consol or terminal if verbose is True
        
    display : boolean, optional, default is False
        Display the signal if True
              
    Returns
    -------
    bcn : float
        The background noise level 
    """        
    
    if X.ndim ==2: 
        if axis == 0:
            X = X.transpose()
            axis = 1
    elif X.ndim ==1: 
        axis = 0
        
    if mode=='ale':
                
        if X.ndim ==2:
            bgn = []
            for i, x in enumerate(X):  
                # Min and Max of the envelope (without taking into account nan)
                x_min = np.nanmin(x)
                x_max = np.nanmax(x)
                # Compute a 50-bin histogram ranging between Min and Max values
                hist, bin_edges = np.histogram(x, bins=50, range=(x_min, x_max))
                
                # smooth the histogram
                #kernel = np.ones(5)/5
                #hist = np.convolve(hist, kernel, mode='same')
                kernel = ('boxcar', 7)
                hist = fir_filter(hist,kernel, axis=axis)
                   
                if display:
                    # Plot only the first histogram
                    #if i == 0 :
                        #n, bins, patches = plt.hist(x=s, bins=100, color='#0504aa', alpha=0.7, rwidth=0.85)
                        plt.figure()
                        plt.plot(bin_edges[0:-1],hist)
                    
                # find the maximum of the peak with quadratic interpolation
                # don't take into account the first 4 bins.
                imax = np.argmax(hist[4::]) + 4

#                # Check the boundary
#                if (imax <= 4 ) :
#                    bin_edges_interp = bin_edges
#                    hist_interp = 4
#                elif (imax >= (len(hist)-2)) :
#                    bin_edges_interp = bin_edges
#                    hist_interp = len(hist)-2
#                else :
#                    f = interp1d(bin_edges[imax-2:imax+2], hist[imax-2:imax+2], kind='quadratic')
#                    bin_edges_interp = np.arange(bin_edges[imax-1], bin_edges[imax+1], 0.01)
#                    hist_interp = f(bin_edges_interp)   # use interpolation function returned by `interp1d`
#                    if display:
#                        plt.plot(bin_edges_interp,hist_interp)
#                                        
#                # assuming an additive noise model : noise_bckg is the max of the histogram
#                # as it is an histogram, the value is 
#                bgn.append(bin_edges_interp[np.argmax(hist_interp)])
                
                bgn.append(bin_edges[imax])
                
        
            # transpose the vector
            bgn = np.asarray(bgn)
            bgn = bgn.transpose()
        else:
            x = X
            # Min and Max of the envelope (without taking into account nan)
            x_min = np.nanmin(x)
            x_max = np.nanmax(x)
            
            # Compute a 50-bin histogram ranging between Min and Max values
            hist, bin_edges = np.histogram(x, bins=50, range=(x_min, x_max))
            
            # smooth the histogram
            #kernel = np.ones(5)/5
            #hist = np.convolve(hist, kernel, mode='same')
            kernel = ('boxcar', 7)
            hist = fir_filter(hist,kernel, axis=axis)
            
            if display:
                #n, bins, patches = plt.hist(x=s, bins=20, color='#0504aa', alpha=0.7, rwidth=0.85)
                plt.figure()
                plt.plot(bin_edges[0:-1],hist)
                         
            # find the maximum of the peak with quadratic interpolation
            imax = np.argmax(hist)
            
            # Check the boundary
#            if (imax <= 5) or (imax >= len(hist)-5):
#                bin_edges_interp = bin_edges
#                hist_interp = imax
#            else :
#                f = interp1d(bin_edges[imax-2:imax+2], hist[imax-2:imax+2], kind='quadratic')
#                bin_edges_interp = np.arange(bin_edges[imax-1], bin_edges[imax+1], 0.01)
#                hist_interp = f(bin_edges_interp)   # use interpolation function returned by `interp1d`
#            
#            if display:
#                plt.plot(bin_edges_interp,hist_interp)
            
            # assuming an additive noise model : noise_bckg is the max of the histogram
            # as it is an histogram, the value is 
#            bgn = bin_edges_interp[np.argmax(hist_interp)]
            bgn = bin_edges[imax]

    elif mode=='median':
        bgn = median(X, axis=axis)
        
    elif mode=='mean':
        bgn = mean(X, axis=axis)
        
    return bgn

#=============================================================================
def raoQ (p, bins):
    """
        compute Rao's Quadratic entropy in 1d
    """
    
    # be sure they are ndarray
    p = np.asarray(p)
    bins = np.asarray(bins)
    
    # Normalize p by the sum in order to get the sum of p = 1
    p = p/sum(p)
    
    # take advantage of broadcasting, 
    # Get the pairwise distance 
    # Euclidian distance
    d = abs(bins[..., np.newaxis] - bins[np.newaxis, ...])
    # Keep only the upper triangle (symmetric)
    #d = np.triu(d, 0)
        
    # compute the crossproduct of pixels value pi,pj
    pipj = (p[..., np.newaxis] * p[np.newaxis, ...])
    #pipj = np.triu(pipj, 0)
    # Multiply by 2 to take into account the lower triangle (symmetric)
    Q = sum(sum(pipj*d))/len(bins)**2
    
    return Q

#=============================================================================
"""
    Indices based on the entropy
"""
def spectral_entropy (X, fn, frange=None, display=False) :

    
    if isinstance(frange, numbers.Number) :
        print ("WARNING: frange must be a tupple (fmin, fmax) or None")
        return
    
    if frange is None : frange=(fn.min(),fn.max())
    
    # select the indices corresponding to the frequency range
    iBAND = index_bw(fn, frange)

    # TOWSEY & BUXTON : only on the bio band
    """ EAS [TOWSEY] """
    """ VALIDATION with R : OK
        COMMENT : Result a bit different due to different Hilbert implementation
    """
    X_mean = mean(X[iBAND], axis=1)
    Hf = entropy(X_mean)
    EAS = 1 - Hf

    #### Entropy of spectral variance (along the time axis for each frequency)
    """ ECU [TOWSEY] """
    """ VALIDATION with R : OK
    """
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
    """ VALIDATION with R : OK
    """
    ioffset = np.argmax(iBAND==True)
    Nbins = sum(iBAND==True)    
    imax_X = np.argmax(X[iBAND],axis=0) + ioffset
    imax_X = fn[imax_X]
    max_X_bin, bin_edges = np.histogram(imax_X, bins=Nbins, range=frange)
    max_X_bin = max_X_bin/sum(max_X_bin)
    Hf_fmax = entropy(max_X_bin)
    EPS = 1 - Hf_fmax    
    
    #### Kurtosis of spectral maxima
    """ VALIDATION with R : OK
    """
    KURT = kurtosis(max_X_bin)
    
    #### Kurtosis of spectral maxima
    """ VALIDATION with R : OK
    """
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

def index_bw (fn, bw):
    """
    Select the index min and max coresponding to the selected frequency band
    
    example :
    fn = 44100
    bw = (100,1000) #in Hz
    Sxx = Sxx[index_bw(fn, bw),)]
    """
    # select the indices corresponding to the frequency bins range
    if bw is None :
        # index = np.arange(0,len(fn),1)
        index = (np.ones(len(fn)))
        index = [bool(x) for x in index]
    elif isinstance(bw, tuple) :
        #index = (fn>=bw[0]) *(fn<=bw[1])
        index = (fn>=fn[(abs(fn-bw[0])).argmin()]) *(fn<=fn[(abs(fn-bw[1])).argmin()])
    elif isinstance(bw, numbers.Number) : 
        # index =  (abs(fn-bw)).argmin()
        index = np.zeros(len(fn))
        index[(abs(fn-bw)).argmin()] = 1
        index = [bool(x) for x in index]
    return index

#=============================================================================
    
def energy_per_freqbin (PSDxx, fn, frange = (0, 20000), bin_step = 1000):
        
    #Convert into bins
    PSDxx_bins, bins = intoBins(PSDxx, fn, bin_min=0, bin_max=fn[-1], 
                              bin_step=bin_step, axis=0)   
    
    # select the indices corresponding to the frequency bins range
    indf = index_bw (bins, frange) 

    # select the frequency bins and take the min
    energy = sum(PSDxx_bins[indf, ])
    
    return energy

#=============================================================================
def soundscapeIndex (PSDxx,fn,frange_bioPh=(1000,10000),frange_antroPh=(0,1000), 
                     step=None):
    
    # Frequency resolution
    # if step is None, keep the original frequency resolution, otherwise,
    # the spectrogram is converted into new frequency bins
    if step is None : step = fn[1]-fn[0]
    
    # Energy in BIOBAND
    bioPh = energy_per_freqbin(PSDxx, fn, frange=frange_bioPh, bin_step=step)
    # Energy in ANTHROPOBAND
    antroPh = energy_per_freqbin(PSDxx, fn, frange=frange_antroPh, bin_step=step)
    
    # NDSI and ratioBA indices 
    NDSI = (bioPh-antroPh)/(bioPh+antroPh)
    ratioBA = bioPh / antroPh
    
    return NDSI, ratioBA, antroPh, bioPh

#=============================================================================
def bioacousticsIndex (SxxdB, fn, frange=(2000, 15000)):
    """
    Voir la fonction de soundecology sous R.
    2 trucs bizarre :
        - Conversion en dB en normalisant par le max (à éviter)
        - multiplication par df sans diviser par la bande passante (=longueur) ce qui ne donne pas une aire...
    """    
    # ======= As soundecology
    # Convert into dB by normalizing with the max
    #SxxdB = 20*log10(Sxx/max(Sxx))
    
    # Mean dB spectrum
    Sxx = dB2linear(SxxdB)
    meanSxx = mean(Sxx, axis=1)  
    meanSxxdB = linear2dB(meanSxx)
    # select the indices corresponding to the frequency bins range
    indf = index_bw(fn,frange)
    # "normalization" in order to get positive 'vectical' values 
    meanSxxdB = meanSxxdB[indf,]-min(meanSxxdB[indf,])
    
    # Compute the area using the composite trapezoidal rule.
    # frequency resoluton. 
    # Normalize by the frequency range in order to get unitless result
    df = fn[1] - fn[0]
    BI = np.trapz(meanSxxdB, dx=df) / (frange[1]-frange[0])
    # or other method to calculate the area under the curve 
    #BI = sum(meanSxxdB)*df / (frange[1]-frange[0])
    # Very similar to basic average...    
    #BI = mean(meanSxxdB)
    
    # ======= As soundecology
    # Small difference in the result due to different values of the spectrogram...
    #BI = sum(meanSxxdB)*df
    
    return BI
    
"""
    Indices based on the acoustic event
"""

def acoustic_activity (xdB, dB_threshold, axis=1):
    # ACTsp [Towsey] : ACTfract (proportion (fraction) of point value above the theshold)
    # EVNsp [Towsey] : ACTcount (number of point value above the theshold)
    ACTfract, ACTcount = score(xdB, dB_threshold, axis=axis)
    ACTfract= ACTfract.tolist()
    ACTcount = ACTcount.tolist()
    ACTmean = mean(dB2linear(xdB[xdB>dB_threshold]))
    return ACTfract, ACTcount, ACTmean
       

#=============================================================================     
def acoustic_events(xdB, dB_threshold, dt, rejectLength=None):
    # EVNsum : total events duration (s) 
    # EVNmean : mean events duration (s)
    # EVNcount : number of events per s
    
    # total duration
    if xdB.ndim ==1 : duration = (len(xdB)-1) * dt
    if xdB.ndim ==2 : duration = (xdB.shape[1]-1) * dt
    
    xdB = np.asarray(xdB)
    # thresholding => binary
    EVN = (xdB>=dB_threshold)*1  
    # Remove events shorter than 'rejectLength' 
    # (done by erosion+dilation = opening)
    if rejectLength is not None:
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



from scipy.io import wavfile 
def load(filename, channel='left', detrend=True, verbose=False,
         display=False, savefig=None, **kwargs): 

    if verbose :
        print(72 * '_' )
        print("loading %s..." %filename)   
    
    # read the .wav file and return the sampling frequency fs (Hz) 
    # and the audiogram s as a 1D array of integer
    fs, s = wavfile.read(filename)
    if verbose :print("Sampling frequency: %dHz" % fs)
    
    # Normalize the signal between -1 to 1 depending on the type
    if s.dtype == np.int32:
        bit = 32
        s = s/2**(bit-1)
    elif s.dtype == np.int16:
        bit = 16
        s = s/2**(bit-1)
    elif s.dtype == np.uint8:
        bit = 8
        s = s/2**(bit) # as it's unsigned
    
    # test if stereo signal. if YES => keep only the ch_select
    if s.ndim==2 :
        if channel == 'left' :
            if verbose :print("Select left channel")
            s_out = s[:,0] 
        else:
            if verbose :print("Select right channel")
            s_out = s[:,1] 
    else:
        s_out = s;
        
    # Detrend the signal by removing the DC offset
    if detrend: s_out = s_out - mean(s_out)
                               
    return s_out, fs

#=============================================================================

"""
    Indices based on the texture (grey-level co-occurrence matrix (GLCM))
"""


"""
===============================================================================
 
                     TEST
                     
===============================================================================                 
"""

if TEST == True:
    
    CHANNEL = 'right'
    MODE_SPECTRO = 'amplitude'  # 'psd'  #'amplitude'
    MODE_ENV = 'hilbert'        # 'fast' #'hilbert'
    
    N = 512     # frame size (in points)
    WAVE_DURATION = 60 # duration of the recording in seconds
    
    MIN_dB = 120
    dB_GAIN = 0
    
    FREQ_ANTHRO_MIN = 0
    FREQ_ANTHRO_MAX = 1000
    FREQ_BIO_MIN = 1000
    FREQ_BIO_MAX = 15000
    FREQ_INSECT_MIN = 15000
    FREQ_INSECT_MAX = 20000
    
    ANTHRO_BAND = (FREQ_ANTHRO_MIN, FREQ_ANTHRO_MAX)
    BIO_BAND = (FREQ_BIO_MIN,FREQ_BIO_MAX)
    INSECT_BAND = (FREQ_INSECT_MIN,FREQ_INSECT_MAX)
    
    DISPLAY = True
    
    #fullfilename = '/home/haupert/DATA/mes_projets/_TOOLBOX/Python/maad_project/scikit-maad/data/guyana_tropical_forest.wav'
    fullfilename = '/home/haupert/DATA/mes_projets/_TOOLBOX/Python/maad_project/scikit-maad/data/jura_cold_forest.wav'
    
    """===========================================================================
    ==============================================================================
                     Computation in the time domain 
    ==============================================================================
     ========================================================================== """
                     
    #### Load the original sound
    wave,fs = load(filename=fullfilename, channel=CHANNEL, detrend=False, verbose=False)

    #### Highpass signal (10Hz)
    wave = iir_filter1d(wave,fs,fcut=10,forder=10,fname='butter',ftype='highpass')

    """ ==========================================================================
    ==============================================================================
                     Computation in the frequency domain 
    ==============================================================================
    =========================================================================="""
 
    #### spectrogram => mode : 'amplitude' or 'psd'
    Axx,tn,fn = spectrogram(wave, fs, N=N, mode=MODE_SPECTRO, detrend=False, verbose=False) 
        
    if MODE_SPECTRO == 'psd':
        Sxx = sqrt(Axx)
        PSDxx = Axx
        #### convert into dB
        SxxdB = linear2dB(Sxx, db_range=MIN_dB, db_gain=dB_GAIN)
    elif MODE_SPECTRO =='amplitude':
        Sxx = Axx
        PSDxx = Axx**2
        #### convert into dB
        SxxdB = linear2dB(Sxx, db_range=MIN_dB, db_gain=dB_GAIN)
            
    # display full SPECTROGRAM in dB
    if DISPLAY :
        
        fig4, ax4 = plt.subplots()
        # set the paramteers of the figure
        fig4.set_facecolor('w')
        fig4.set_edgecolor('k')
        fig4.set_figheight(4)
        fig4.set_figwidth (13)
                
        # display image
        _im = ax4.imshow(SxxdB, extent=(tn[0], tn[-1], fn[0], fn[-1]), 
                         interpolation='none', origin='lower', 
                         vmin =-MIN_dB, vmax=0, cmap='gray')
        plt.colorbar(_im, ax=ax4)
 
        # set the parameters of the subplot
        ax4.set_title('Spectrogram')
        ax4.set_xlabel('Time [sec]')
        ax4.set_ylabel('Frequency [Hz]')
        ax4.axis('tight') 
     
        fig4.tight_layout()
         
        # Display the figure now
        plt.show()
     
        # display MEAN SPECTROGRAM in dB [anthropological and Biological bands]
        if DISPLAY :
            fig5, ax5 = plt.subplots()
            ax5.plot(fn[index_bw(fn,ANTHRO_BAND)], 
                        mean(SxxdB[index_bw(fn,ANTHRO_BAND)], axis=1), color='#555555', lw=2, alpha=1)
            ax5.plot(fn[index_bw(fn,BIO_BAND)], 
                        mean(SxxdB[index_bw(fn,BIO_BAND)], axis=1), color='#55DD00', lw=2, alpha=1)
            ax5.plot(fn[index_bw(fn,INSECT_BAND)], 
                        mean(SxxdB[index_bw(fn,INSECT_BAND)], axis=1), color='#DDDC00', lw=2, alpha=1)        


