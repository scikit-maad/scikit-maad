#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt
import numpy as np 
from numpy import log10, diff, mean
import pandas as pd
import numbers
import warnings

# min value
import sys
_MIN_ = sys.float_info.min

#=============================================================================

def index_bw (fn, bw):
    """
    Select all the index coresponding to the selected frequency band
    
    Parameters
    ----------
    fn :  1d ndarray of scalars
        Vector of frequencies
        
    bw : single value or tupple of two values
        if single value : frequency to select
        if tupple of two values : min frequency and max frequency to select
        
    Returns
    -------
    index : 1d ndarray of scalars
        Vector of booleans corresponding to the selected frequency(-ies)
    
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> PSDxx,tn,fn,_ = maad.sound.spectrogram(w,fs,window='hanning',noverlap=512, nFFT=1024)
    >>> PSDxxdB = maad.util.linear2dB (PSDxx) # convert into dB
    >>> bw = (2000,6000) #in Hz
    >>> fig_kwargs = {'vmax': max(PSDxxdB),
                      'vmin':min(PSDxxdB),
                      'extent':(tn[0], tn[-1], fn[0], fn[-1]),
                      'figsize':(10,13),
                      'title':'Power Spectrum Density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    >>> maad.util.plot2D(PSDxxdB,**fig_kwargs)
    >>> PSDxxdB_crop = PSDxxdB[index_bw(fn, bw)]
    >>> fn_crop = fn[index_bw(fn, bw)]
    >>> fig_kwargs = {'vmax': max(PSDxxdB),
                      'vmin':min(PSDxxdB),
                      'extent':(tn[0], tn[-1], fn_crop[0], fn_crop[-1]),
                      'figsize':(10*len(fn_crop)/len(fn),13),
                      'title':'Power Spectrum Density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    >>> maad.util.plot2D(PSDxxdB_crop,**fig_kwargs)
    
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
        raise Exception('WARNING: bin step must be larger or equal than the actual resolution of x')

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
       
def rle(x):
    """
    Run-Length encoding    
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
    >>> a = np.array([1,2,3,4,5])
    >>> linear_scale(a, 0, 1)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        
    References
    ----------
    Written by Aniruddha Kembhavi, July 11, 2007 for MATLAB
    Adapted by S. Haupert Dec 12, 2017 for Python
    """

    # if x is a list, convert x into ndarray 
    if isinstance(x, list):
        x = np.asarray(x)
        
    # avoid problems with inf and -inf values
    x_min = np.min(x.ravel()[x.ravel()!=-np.inf]) 
    x[np.where(x == -np.inf)] = x_min
    x_max = np.max(x.ravel()[x.ravel()!=np.inf]) 
    x[np.where(x == np.inf)] = x_max        
     
    # do the normalization
    y = x - x.min();
    y = (y/y.max())*(maxval-minval);
    y = y + minval;
    return y


#=============================================================================
def amplitude2dB (x, db_range=None, db_gain=0):
    """
    Transform amplitude data (signal, scalar) into decibel scale within the dB range (db_range).
    A gain (db_gain) could be added at the end.    
    
    Parameters
    ----------
    x : array-like or scalar
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
        y = 20*log10(x) + db_gain  
        
    Examples
    --------
    >>> a = np.array([1,2,3,4,5])
    >>> amplitude2dB(a, mode='amplitude')
        array([ 0.        ,  6.02059991,  9.54242509, 12.04119983, 13.97940009])
        
    """            
    x = abs(x)   # take the absolute value of x
    
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
def power2dB (x, db_range=None, db_gain=0):
    """
    Transform power (amplitude²) signal or scalar into decibel scale within the dB range (db_range).
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
        y = 10*log10(x) + db_gain  
        
    Examples
    --------
    >>> a = np.array([1,2,3,4,5])
    >>> power2dB(a**2, mode='power')
        array([ 0.        ,  6.02059991,  9.54242509, 12.04119983, 13.97940009])
        
    """            
    x = abs(x)   # take the absolute value of x
    
    # Avoid zero value for log10  
    # if it's a scalar
    if hasattr(x, "__len__") == False:
        if x ==0: x = _MIN_  
    else :     
        x[x ==0] = _MIN_  # Avoid zero value for log10 

    # conversion in dB  
    y = 10*log10(x)   # take log 
    
    if db_gain : y = y + db_gain    # Add gain if needed
    
    if db_range is not None :
        # set anything above db_range as 0
        y[y > 0] = 0  
        # set anything less than -db_range as -db_range
        y[y < -(db_range)] = -db_range  
        
    return y

#=============================================================================
def dB2amplitude (x, db_gain=0):
    """
    Transform data in dB scale into amplitude
    A gain (db_gain) could be added at the end.    
    
    Parameters
    ----------
    x : array-like
        data in dB to rescale in amplitude 
    db_gain : scalar, optional, default is 0
        Gain that was added to the result 
        --> 20*log10(x) + db_gain
                
    Returns
    -------
    y : scalars
        output in amplitude unit
        
    Examples
    --------
    >>> a = np.array([ 0.        ,  6.02059991,  9.54242509, 12.04119983, 13.97940009])
    >>> dB2amplitude(a)
        array([1., 2., 3., 4., 5.])
        
    """  
    y = 10**((x- db_gain)/20) 

    return y

#=============================================================================
def dB2power (x, db_gain=0):
    """
    Transform data in dB scale into power (amplitude²)
    A gain (db_gain) could be added at the end.    
    
    Parameters
    ----------
    x : array-like
        data in dB to rescale in power 
    db_gain : scalar, optional, default is 0
        Gain that was added to the result 
        --> 10*log10(x) + db_gain
                
    Returns
    -------
    y : scalars
        output in power unit
        
    Examples
    --------
    >>> a = np.array([ 0.        ,  6.02059991,  9.54242509, 12.04119983, 13.97940009])
    >>> dB2power(a)
        array([ 1.        ,  4.        ,  8.99999999, 16.00000001, 25.00000002])        
    """  
    y = 10**((x- db_gain)/10) 
     
    return y

#=============================================================================
def add_dB(*argv, axis=0): 
    """
    add dB values 
        
    Parameters
    ----------
    *argv : ndarray-like of floats
        Arrays containing the sound waveform in dB 
                
    axis : integer, optional, default is 0
        if addition of multiple arrays, select the axis on which the sum is done
                
    Returns
    -------
    e_sum : ndarray-like of floats
        Array containing the sum of the dB values
        
    Examples
    --------
    
    Example with an audio file
    
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> PSDxx,tn,fn,_ = maad.sound.spectrogram(w,fs)
    >>> L = maad.util.power2dBSPL(PSDxx,gain=42)
    >>> L_sum = maad.util.add_dB(L, axis=2)  
    >>> fig_kwargs = {'figtitle':'Spectrum (PSD)',
                      'xlabel':'Frequency [Hz]',
                      'ylabel':'Power [dB]',
                      }
    >>> fig, ax = maad.util.plot1D(fn, L_sum.transpose(), **fig_kwargs)

    Example with single values
    
    >>> L1 = 90 # 90dB
    >>> maad.util.add_dB(L1,L1)
        93.01029995663981
        
    Example with arrays
    
    >>> L1 = [90,80,70]
    >>> maad.util.add_dB(L1,L1,axis=1)
        array([90.45322979, 90.45322979])
    >>> maad.util.add_dB(L1,L1,axis=0)
        array([93.01029996, 83.01029996, 73.01029996])
    """    
    # force to be ndarray
    L = np.asarray(argv)
    
    # test if the 1st dim is = 1 (in case argv is a single 2d array) in order to remove this dimension
    if L.shape[0] == 1 :
        L = L[0,]
   
    # Verify the adequation between axis number and number of dimensions of L
    if axis >= L.ndim:
        axis= L.ndim -1
    
    # dB to energy as sum has to be done with energy
    e = dB2power(L)
    e_sum = e.sum(axis)
    
    
    # test if 2 dimensions but length on 1 dimension is 1
    if e_sum.ndim == 2 :
        if (e_sum.shape[0] ==1) or (e_sum.shape[1] ==1) : 
            e_sum = np.ndarray.flatten(e_sum)
    
    # energy=>pressure to dB
    L_sum = power2dB(e_sum)
        
    return L_sum

#=============================================================================
def mean_dB(*argv, axis=0): 
    """
    Compute the average of dB values
        
    Parameters
    ----------
    *argv : ndarray-like of floats
        Arrays containing the sound waveform in dB
                
    axis : integer, optional, default is 0
        if addition of multiple arrays, select the axis on which the sum is done
                
    Returns
    -------
    e_mean : ndarray-like of floats
        Array containing the mean of the dB values
        
    Examples
    --------
    
    Example with an audio file
    
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> Pxx,tn,fn,_ = maad.sound.spectrogram(w,fs)
    >>> L = maad.util.power2dBSPL(PSDxx,gain=42)
    >>> L_mean = maad.util.mean_dB(L, axis=2)  
    >>> fig_kwargs = {'figtitle':'Power spectrum (PSD)',
                      'xlabel':'Frequency [Hz]',
                      'ylabel':'Power [dB]',
                      }
    >>> fig, ax = maad.util.plot1D(fn, L_mean.transpose(), **fig_kwargs)

    Example with single values
    
    >>> L1 = 90 # 90dB
    >>> L2 = 96 # 96dB
    >>> maad.util.mean_dB(L1,L2)
        93.96292
        
    Example with arrays
    
    >>> L1 = [90,80,70]
    >>> maad.util.mean_dB(L1,L1)
        array([85.68201724, 85.68201724])
    >>> maad.util.mean_dB(L1,L1, axis=0)  
        array([90., 80., 70.]) 
        
    """    
    # force to be ndarray
    L = np.asarray(argv)
    
    # test if the 1st dim is = 1 (in case argv is a single 2d array) in order to remove this dimension
    if L.shape[0] == 1 :
        L = L[0,]
    
    # Verify the adequation between axis number and number of dimensions of L
    if axis >= L.ndim:
        axis= L.ndim -1
    
    # dB to energy as sum has to be done with energy
    e = dB2power(L)
    e_mean = e.mean(axis)
    
    # energy (power) => dB
    e_mean = power2dB(e_mean)
    
    # test if 2 dimensions but length on 1 dimension is 1
    if e_mean.ndim == 2 :
        if (e_mean.shape[0] ==1) or (e_mean.shape[1] ==1) : 
            e_mean = np.ndarray.flatten(e_mean)
    
    # test if it's a single value
    if e_mean.size == 1 : 
        e_mean = float(e_mean)
    
    return e_mean

#=============================================================================

def shift_bit_length(x):
    """
    find the closest power of 2 that is superior or equal to the number x
    
    Parameters
    ----------
    x : scalar
    
    Returns
    -------
    y : scalar
        the closest power of 2 that is superior or equal to the number x
    
    Examples
    --------
    >>> shift_bit_length(1000)
        1024
    """
    y = 1<<(x-1).bit_length()
    return y

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
    >>> nearest_idx(x, 1.3)
        0
    >>> nearest_idx(x, 1.6)
        1
    """
    # be sure it's ndarray
    array = np.asarray(array)
    
    idx = (np.abs(array-value)).argmin()
    return idx

#=============================================================================

def get_df_single_row (df, index, mode='iloc'):
    """
    Extract a single row from a dataframe keeping the DataFrame type (instead of becoming a Series)
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with one or more rows
    
    index : interger (in case of 'iloc') or index (in case of loc)
        index could be the row number or the row index depending on the mode
    
    mode : string, optional, default is 'iloc'
        choose between row number ('iloc') or row index ('loc')
        (see Pandas documentation for the difference)
    Returns
    -------
    df_out : pandas DataFrame
        Single row DataFrame
    
    Examples
    --------
    >>>import pandas as pd
    >>>from maad.util import get_df_single_row
    >>>species = [ ('bird', 8, 'Yes' ,'NYC') ,
                  ('insect', 4, 'No','NYC' ) ,
                  ('mammal', 2, 'No','NYC' ) ,
                  ('frog', 3, 'Yes',"LA" ) ]
    
    >>>df = pd.DataFrame(species, 
                          columns = ['category',
                                     'abundance',
                                     'presence',
                                     'location']) 
    >>>print(df)
          category  abundance presence location
        0     bird          8      Yes      NYC
        1   insect          4       No      NYC
        2   mammal          2       No      NYC
        3     frog          3      Yes       LA
    >>> df.loc[0]
        category     bird
        abundance       8
        presence      Yes
        location      NYC
        Name: 0, dtype: object
    >>> get_df_single_row (df, index=0, mode='iloc')
          category abundance presence location
        0     bird         8      Yes      NYC
        
     Now with index   
        
    >>>df_modified=df.set_index("category")
    >>>print(df_modified)
                    abundance presence location
        category                             
        bird              8      Yes      NYC
        insect            4       No      NYC
        mammal            2       No      NYC
        frog              3      Yes       LA 
    >>> df_modified.loc['insect']  
        abundance      4
        presence      No
        location     NYC
        Name: insect, dtype: object
    >>> get_df_single_row (df_modified, index='insect', mode='loc')
               abundance presence location
        insect         4       No      NYC
    >>> get_df_single_row (df_modified, index=1, mode='iloc')    
               abundance presence location
        insect         4       No      NYC
    """

    if mode == 'iloc' :
        df_out = pd.DataFrame(df.iloc[index]).T
    elif mode == 'loc' :
        df_out = pd.DataFrame(df.loc[index]).T
    else :
        raise TypeError ('mode must be iloc or loc, default is iloc')
    
    return df_out
    

def format_features(df, tn, fn):
    """ 
    Setup rectangular df to a predifined format: 
    time-frequency or bounding box
    
    Parameters
    ----------
    df : pandas DataFrame
        array must have a valid input format with column names
        
        - bounding box: min_y, min_x, max_y, max_x
        - time frequency: min_f, min_t, max_f, max_t
        - centroid tf : centroid_f, centroid_t
        - centroid pixels : centroid_y, centroid_x
    
    tn : ndarray
        vector with temporal indices, output from the spectrogram function (in seconds)
    fn: ndarray
        vector with frequencial indices, output from the spectrogram function (in Hz)

    Returns
    -------
    df: ndarray
        array with indices of ROIs matched on spectrogram
    """
    # Check format of the input data
    if type(df) is not pd.core.frame.DataFrame :
        raise TypeError('Rois must be of type pandas DataFrame')  

    if ('min_t' and 'min_f' and 'max_t' and 'max_f') in df and not (('min_y' and 'min_x' and 'max_y' and 'max_x') in df):
        
        # Check that time and frequency limits are inside tn and fn limits
        # df = df[(df.min_t >= tn.min()) & (df.max_t <= tn.max()) & (df.min_f >= fn.min()) & (df.max_f <= fn.max())]
        if (df.min_t < tn.min()).any() | (df.max_t > tn.max()).any() | (df.min_f < fn.min()).any() | (df.max_f > fn.max()).any():
            warnings.warn('ROIs boundaries are outside time or frecuency signal limits. Clipping ROIS to valid boundaries.')
            df.loc[df.min_t < tn.min(),'min_t'] = tn.min()
            df.loc[df.max_t > tn.max(),'max_t'] = tn.max()
            df.loc[df.min_f < fn.min(),'min_f'] = fn.min()
            df.loc[df.max_f > fn.max(),'max_f'] = fn.max()
        
        df_bbox = []
        for idx in df.index:            
            min_y = nearest_idx(fn, df.loc[idx, 'min_f'])
            min_x = nearest_idx(tn, df.loc[idx, 'min_t'])
            max_y = nearest_idx(fn, df.loc[idx, 'max_f'])
            max_x = nearest_idx(tn, df.loc[idx, 'max_t'])
            # test if max > min, if not, drop row form df
            if (min_y<=max_y) and (min_x<=max_x) :
                df_bbox.append((min_y, min_x, max_y, max_x))
            else: 
                df = df.drop(idx)
        
        df = df.join(pd.DataFrame(df_bbox, 
                                  columns=['min_y','min_x','max_y','max_x'], 
                                  index=df.index))
                        
    if ('min_y' and 'min_x' and 'max_y' and 'max_x') in df and not (('min_t' and 'min_f' and 'max_t' and 'max_f') in df): 
        df_bbox = []
        for _,row in df.iterrows():            
            min_f = fn[int(round(row.min_y))]
            min_t = tn[int(round(row.min_x))]
            max_f = fn[int(round(row.max_y))]
            max_t = tn[int(round(row.max_x))]
            # test if max > min, if not, drop row form df
            if (min_f<=max_f) and (min_t<=max_t) :
                df_bbox.append((min_f, min_t, max_f, max_t))
            else: 
                df = df.drop(row.name)

        df = df.join(pd.DataFrame(df_bbox, 
                                  columns=['min_f','min_t','max_f','max_t'], 
                                  index=df.index))
    
    if ('centroid_y' and 'centroid_x') in df and not (('centroid_f' and 'centroid_t') in df): 
        df_centroid = []
        for _,row in df.iterrows():            
            centroid_f = fn[int(round(row.centroid_y))]
            centroid_t = tn[int(round(row.centroid_x))]
            df_centroid.append((centroid_f, centroid_t))
            
        df = df.join(pd.DataFrame(df_centroid, 
                                      columns=['centroid_f','centroid_t'], 
                                      index=df.index)) 
        
    if ('centroid_f' and 'centroid_t') in df and not (('centroid_y' and 'centroid_x') in df) : 
        df_centroid = []
        for idx in df.index:            
            centroid_y = nearest_idx(fn, df.loc[idx, 'centroid_f'])
            centroid_x = nearest_idx(tn, df.loc[idx, 'centroid_t'])
            df_centroid.append((centroid_y, centroid_x))
            
        df = df.join(pd.DataFrame(df_centroid, 
                                      columns=['centroid_y','centroid_x'], 
                                      index=df.index))     
     
    #=============
    if ('duration_x' and 'bandwidth_y' and 'area_xy') in df and not (('duration_t' and 'bandwidth_f' and 'area_tf') in df): 
        df_area = []
        for _,row in df.iterrows():            
            bandwidth_f = row.bandwidth_y * (fn[1]-fn[0])
            duration_t = row.duration_x * (tn[1]-tn[0])
            area_tf = row.area_xy * (fn[1]-fn[0]) * (tn[1]-tn[0])
            df_area.append((duration_t, bandwidth_f, area_tf))
            
        df = df.join(pd.DataFrame(df_area, 
                                  columns=['duration_t','bandwidth_f', 'area_tf'], 
                                  index=df.index)) 
        
    if ('duration_t' and 'bandwidth_f' and 'area_tf') in df and not (('duration_x' and 'bandwidth_y' and 'area_xy') in df) : 
        df_area = []
        for idx in df.index:            
            bandwidth_y = round(df.loc[idx, 'bandwidth_f']) / ((fn[1]-fn[0]))
            duration_x = round(df.loc[idx, 'duration_t']) / (tn[1]-tn[0])
            area_xy = round(df.loc[idx, 'area_xy'] / ((fn[1]-fn[0]) * (tn[1]-tn[0])))
            df_area.append((bandwidth_y, duration_x, area_xy))
            
        df = df.join(pd.DataFrame(df_area, 
                                  columns=['duration_x','bandwidth_y','area_xy'], 
                                  index=df.index))  
                
    return df