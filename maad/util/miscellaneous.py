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
from numpy import log10, diff
import pandas as pd
import numbers

# min value
import sys
_MIN_ = sys.float_info.min

#=============================================================================

def index_bw (fn, bw):
    """
    Select the index min and max coresponding to the selected frequency band
    
    Examples
    --------
    >>> fn = 44100
    >>> bw = (100,1000) #in Hz
    >>> X = X[index_bw(fn, bw)]
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

def linear_scale(x, minval= 0.0, maxval=1.0):
    """ 
    Program to scale the values of a matrix from a user specified minimum to 
    a user specified maximum
    
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
        
    Examples
    --------
    >>> a = np.array([1,2,3,4,5]);
    >>> a_out = scaledata(a,0,1); 
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        
    References
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
def linear2dB (x, mode = 'power', db_range=None, db_gain=0):
    """
    Transform linear date into decibel scale within the dB range (db_range).
    A gain (db_gain) could be added at the end.    
    
    Parameters
    ----------
    x : array-like
        data to rescale in dB  
    mode : str, default is 'power'
        select the type of data : 'amplitude' or 'power' to compute the corresponding dB
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
    if mode == 'amplitude' :
        y = 20*log10(x)   # take log
    elif mode == 'power':
        y = 10*log10(x)   # take log 
    
    if db_gain : y = y + db_gain    # Add gain if needed
    
    if db_range is not None :
        # set anything above db_range as 0
        y[y > 0] = 0  
        # set anything less than -db_range as -db_range
        y[y < -(db_range)] = -db_range  
        
    return y

#=============================================================================
def dB2linear (x, mode = 'power',  db_gain=0):
    """
    Transform linear date into decibel scale within the dB range (db_range).
    A gain (db_gain) could be added at the end.    
    
    Parameters
    ----------
    x : array-like
        data in dB to rescale in linear 
    mode : str, default is 'power'
        select 'amplitude' or 'power' to compute the corresponding dB
    db_gain : scalar, optional, default is 0
        Gain that was added to the result 
        --> 20*log10(x) + db_gain
                
    Returns
    -------
    y : scalars
        output in amplitude or power unit
    """  
    if mode == 'amplitude' :
        y = 10**((x- db_gain)/20) 
    elif mode == 'power':
        y = 10**((x- db_gain)/10) 
    return y

#=============================================================================
def rois_to_audacity(fname, onset, offset):
    """ 
    Write audio segmentation to file (Audacity format)
    
    Parameters
    ----------
    fname: str
        filename to save the segmentation
    onset: int, float array_like
        output of a detection method (e.g. find_rois_1d)
    offset: int, float array_like
        output of a detection method (e.g. find_rois_1d)
            
    Returns
    -------
    Returns a csv file
    """
    if onset.size==0:
        print(fname, '< No detection found')
        df = pd.DataFrame(data=None)
        df.to_csv(fname, sep=',',header=False, index=False)
    else:
        label = range(len(onset))
        rois_tf = pd.DataFrame({'t_begin':onset, 't_end':offset, 'xlabel':label})
        rois_tf.to_csv(fname, index=False, header=False, sep='\t') 

#=============================================================================

def rois_to_imblobs(im_blobs, rois_bbox):
    """ 
    Add rois to im_blobs 
    """
    # roi to image blob
    for min_y, min_x, max_y, max_x in rois_bbox.values:
        im_blobs[min_y:max_y+1, min_x:max_x+1]=1
    return im_blobs

#=============================================================================

def shift_bit_length(x):
    """
    find the closest power of 2 that is superior or equal to the number x
    """
    return 1<<(x-1).bit_length()

#=============================================================================
    
def nearest_idx(array,value):
    """ 
    Find nearest value on array and return index
    
    Parameters
    ----------
    array: ndarray
        array of values to search for nearest values
    value: float
        value to be searched in array
            
    Returns
    -------
    idx: int
        index of nearest value on array
        
    Examples
    --------
    >>> x = np.array([1,2,3])
    >>> ig.nearest_idx(x, 1.3)
    [0]
    >>> ig.nearest_idx(x, 1.6)
    [1]
    """
    idx = (np.abs(array-value)).argmin()
    return idx

#=============================================================================
def normalize_2d(im, min_value, max_value):
    """ 
    Normalize 2d array between two values. 
        
    Parameters
    ----------
    im: 2D ndarray
        Bidimensinal array to be normalized
    min_value: int, float
        Minimum value in normalization
    max_value: int, float
        Maximum value in normalization
        
    Returns
    -------
    im_out: 2D ndarray
        Array normalized between min and max values
    """
    # avoid problems with inf and -inf values
    min_im = np.min(im.ravel()[im.ravel()!=-np.inf]) 
    im[np.where(im == -np.inf)] = min_im
    max_im = np.max(im.ravel()[im.ravel()!=np.inf]) 
    im[np.where(im == np.inf)] = max_im

    # normalize between min max
    im = (im - min_im) / (max_im - min_im)
    im_out = im * (max_value - min_value) + min_value
    return im_out

#=============================================================================

def format_rois(rois, ts, fs, fmt=None):
    """ 
    Setup rectangular rois to a predifined format: 
    time-frequency or bounding box
    
    Parameters
    ----------
    rois : pandas DataFrame
        array must have a valid input format with column names
        
        - bounding box: min_y, min_x, max_y, max_x
        
        - time frequency: min_f, min_t, max_f, max_t
    
    ts : ndarray
        vector with temporal indices, output from the spectrogram function (in seconds)
    fs: ndarray
        vector with frequencial indices, output from the spectrogram function (in Hz)
    fmt: str
        A string indicating the desired output format: 'bbox' or 'tf'
        
    Returns
    -------
    rois_bbox: ndarray
        array with indices of ROIs matched on spectrogram
    """
    # Check format of the input data
    if type(rois) is not pd.core.frame.DataFrame and type(rois) is not pd.core.series.Series:
        raise TypeError('Rois must be of type pandas DataFrame or Series.')    

    elif fmt is not 'bbox' and fmt is not 'tf':
        raise TypeError('Format must be either fmt=\'bbox\' or fmt=\'tf\'.')

    # Compute new format
    elif type(rois) is pd.core.series.Series and fmt is 'bbox':
        min_y = nearest_idx(fs, rois.min_f)
        min_x = nearest_idx(ts, rois.min_t)
        max_y = nearest_idx(fs, rois.max_f)
        max_x = nearest_idx(ts, rois.max_t)
        rois_out = pd.Series({'min_y': min_y, 'min_x': min_x, 
                              'max_y': max_y, 'max_x': max_x})
        
    elif type(rois) is pd.core.series.Series and fmt is 'tf':
        rois_out = pd.Series({'min_f': fs[rois.min_y.astype(int)], 
                              'min_t': ts[rois.min_x.astype(int)],
                              'max_f': fs[rois.max_y.astype(int)],
                              'max_t': ts[rois.max_x.astype(int)]})
            
    elif type(rois) is pd.core.frame.DataFrame and fmt is 'bbox':
        rois_bbox = []
        for idx in rois.index:            
            min_y = nearest_idx(fs, rois.loc[idx, 'min_f'])
            min_x = nearest_idx(ts, rois.loc[idx, 'min_t'])
            max_y = nearest_idx(fs, rois.loc[idx, 'max_f'])
            max_x = nearest_idx(ts, rois.loc[idx, 'max_t'])
            rois_bbox.append((min_y, min_x, max_y, max_x))
        
        rois_out = pd.DataFrame(rois_bbox, 
                                columns=['min_y','min_x','max_y','max_x'])
    
    elif type(rois) is pd.core.frame.DataFrame and fmt is 'tf':
        rois_out = pd.DataFrame({'min_f': fs[rois.min_y.astype(int)], 
                                 'min_t': ts[rois.min_x.astype(int)],
                                 'max_f': fs[rois.max_y.astype(int)],
                                 'max_t': ts[rois.max_x.astype(int)]})

    else:
        raise TypeError('Rois type or format not understood, please check docstring.')
    return rois_out
