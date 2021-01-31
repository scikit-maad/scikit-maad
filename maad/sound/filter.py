#!/usr/bin/env python
""" 
Collection of functions to filter audio signal in time and frequency
domains. 
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
import matplotlib.pyplot as plt
from scipy.signal import (sosfiltfilt, convolve, iirfilter, get_window, 
                          kaiserord, firwin, lfilter)
from skimage import filters

# Import internal modules 
from maad.util import plot2d

#%%
# =============================================================================
# public functions
# =============================================================================
def select_bandwidth (x, fs, fcut, forder, fname ='butter', ftype='bandpass', 
                     rp=None, rs=None):
    """
    Use a lowpass, highpass, bandpass or bandstop filter to process a 1d signal with an iir filter.
        
    Parameters
    ----------
    x : array_like
        1d vector of scalars to be filtered
    fs : scalar
        sampling frequency   
    fcut : array_like
        A scalar or length-2 sequence giving the critical frequencies.             
    forder : int
        The order of the filter.
    fname : {`butter`, 'cheby1`, `cheby2`, `ellip`, `bessel`}, optional, default
        is 'butter'   
        The type of IIR filter to design:
        - Butterworth : `butter`
        - Chebyshev I : 'cheby1`
        - Chebyshev II : `cheby2`
        - Cauer/elliptic: `ellip`
        - Bessel/Thomson: `bessel`   
    ftype : {`bandpass`, `lowpass`, `highpass`, `bandstop`}, optional, default 
        is `bandpass`
        The type of filter.
    rp : float, optional
        For Chebyshev and elliptic filters, provides the maximum ripple in 
        the passband. (dB)   
    rs : float, optional
        For Chebyshev and elliptic filters, provides the minimum attenuation 
        in the stop band. (dB)           
            
    Returns
    -------
    y : array_like
        The filtered output with the same shape and phase as x
        
    See Also
    --------
    fir_filter1d
        Lowpass, highpass, bandpass or bandstop a 1d signal with an Fir filter
    
    Examples
    --------
    Load and display the spectrogram of a sound waveform
    
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(w,fs)
    >>> Sxx_dB = maad.util.power2dB(Sxx_power) # convert into dB 
    >>> fig_kwargs = {'vmax': Sxx_dB.max(),
                      'vmin':-90,
                      'extent':(tn[0], tn[-1], fn[0], fn[-1]),
                      'figsize':(4,13),
                      'title':'Power spectrogram density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    >>> fig1, ax1 = maad.util.plot2d(Sxx_dB, **fig_kwargs)
    
    Filter the waveform : keep the bandwidth between 6-10kHz
    
    >>> w_filtered = maad.sound.select_bandwidth(w,fs,fcut=(6000,10000), forder=5, fname ='butter', ftype='bandpass')
    >>> Sxx_power_filtered,tn,fn,_ = maad.sound.spectrogram(w_filtered,fs)
    >>> Sxx_dB_filtered = maad.util.power2dB(Sxx_power_filtered) # convert into dB 
    >>> maad.util.plot2d(Sxx_dB_filtered, **fig_kwargs)
    
    """
    sos = iirfilter(N=forder, Wn=np.asarray(fcut)/(fs/2), btype=ftype,ftype=fname, rp=rp, 
                     rs=rs, output='sos')
    # use sosfiltfilt insteasd of sosfilt to keep the phase of y matches x
    y = sosfiltfilt(sos, x)
    return y
 
#%%
def fir_filter(x, kernel, axis=0):
    """
    Filter a signal using a 1d finite impulse response filter.
    
    This function uses a digital filter based on convolution of 1d kernel over a vector 
    or along an axis of a matrix.
    
    Parameters
    ----------
    x : array_like
        1d vector or 2d matrix of scalars to be filtered
    kernel : array_like or tuple
        Pass directly the kernel (1d vector of scalars) 
        Or pass the arguments in a tuple to create a kernel. Arguments are:   
        - window : string, float, or tuple. The type of window to create. 
        boxcar, triang, blackman, hamming, hann, bartlett, flattop,
        parzen, bohman, blackmanharris, nuttall, barthann, 
        - (kaiser, beta), 
        - (gaussian, standard deviation), 
        - (general_gaussian, power, width), 
        - (slepian, width), 
        - (dpss, normalized half-bandwidth), 
        - (chebwin, attenuation), 
        - (exponential, decay scale), 
        - (tukey, taper fraction)
        - N : length of the kernel
        Examples:
        - kernel = ('boxcar', 9)
        - kernel = (('gaussian', 0.5), 5)
        - kernel = [1 3 5 7 5 3 1] 
    axis : int
        Determine along which axis is performed the filtering in case of 2d matrix
        axis = 0 : vertical
        axis = 1 : horizontal
    
    Returns
    -------
    y : array_like
        The filtered output with the same shape and phase as x
        
    See Also
    --------
    select_bandwidth
        Lowpass, highpass, bandpass or bandstop a 1d signal with an iir filter
    
    Examples
    --------  
    
    Load and display the spectrogram of a sound waveform
    
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> Sxx_power, tn, fn, ext = maad.sound.spectrogram(w,fs)
    >>> Lxx = maad.spl.power2dBSPL(Sxx_power, gain=42) # convert into dB SPL
    >>> fig_kwargs = {'vmax': Lxx.max(),
                      'vmin':0,
                      'extent': ext,
                      'figsize': (4,13),
                      'title': 'Power spectrogram density (PSD)',
                      'xlabel': 'Time [sec]',
                      'ylabel': 'Frequency [Hz]',
                      }
    >>> fig, ax = maad.util.plot2d(Lxx,**fig_kwargs)
    
    Smooth the waveform (lowpass)
    
    >>> w_filtered = maad.sound.fir_filter(w, kernel=(('gaussian', 2), 5))
    >>> Sxx_power_filtered,tn,fn,_ = maad.sound.spectrogram(w_filtered,fs)
    >>> Lxx_filtered = maad.spl.power2dBSPL(Sxx_power_filtered, gain=42) # convert into dB SPL
    >>> fig, ax = maad.util.plot2d(Lxx_filtered,**fig_kwargs)
    
    Smooth the spectrogram, frequency by frequency (blurr)
    
    >>> Lxx_blurr = maad.sound.fir_filter(Lxx, kernel=(('gaussian', 1), 5), axis=1)
    >>> fig, ax = maad.util.plot2d(Lxx_blurr,**fig_kwargs)
    
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

#%%
def sinc(s, cutoff, fs, atten=80, transition_bw=0.05, bandpass=True):
    """
    Filter 1D signal with a Kaiser-windowed filter.
    
    Parameters
    ----------
    s : ndarray
        input 1D signal
    cutoff : ndarray
        upper and lower frequencies (min_f, max_f) in Hertz
    atten : float 
        attenuation in dB
    transition_bw : float
        transition bandwidth in percent default 5% of total band
    bandpass : bool
        bandpass (True) or bandreject (False) filter, default is bandpass
    
    Returns
    -------
    s_filt : array 
        signal filtered
    
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/spinetail.wav')
    >>> import numpy as np
    >>> tn = np.arange(0,len(s))/fs
    >>> s_filt_7_12kHz = maad.sound.sinc(s, cutoff=[7000,12000], fs=fs, atten=80, transition_bw=0.8)
    >>> s_filt_4_8kHz = maad.sound.sinc(s, cutoff=[4500,8000], fs=fs, atten=80, transition_bw=0.8)
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax0, ax1,ax2) = plt.subplots(3,1, sharex=True, squeeze=True)
    >>> maad.util.plot1d(tn,s,ax=ax0, figtitle='original')
    >>> maad.util.plot1d(tn,s_filt_7_12kHz,ax=ax1, figtitle='Kaiser-windowed filter 7-12kHz')
    >>> maad.util.plot1d(tn,s_filt_4_8kHz,ax=ax2, figtitle='Kaiser-windowed filter 4.5-8kHz')
    >>> fig.tight_layout()
    
    """
    width = (cutoff[1] - cutoff[0]) * transition_bw
    numtaps, beta = kaiserord(atten, width/(0.5*fs))
    np.ceil(numtaps-1) // 2 * 2 + 1  # round to nearest odd to have Type I filter
    taps = firwin(numtaps, cutoff, window=('kaiser', beta), 
                         scale=False, nyq=0.5*fs, pass_zero=not(bandpass))
    s_filt = lfilter(taps, 1, s)
    return s_filt

#%%
def smooth (Sxx, std=1, verbose=False, display = False, savefig=None, **kwargs): 
    """ 
    Smooth a spectrogram with a gaussian filter.
     
    Parameters 
    ---------- 
    Sxx : 2d ndarray of scalars 
        Spectrogram (or image) 
     
    std : scalar, optional, default is 1 
        Standard deviation of the gaussian kernel used to smooth the spectrogram.
        The larger is the number, the smoother will be the image and the longer 
        it takes. Standard values should fall between 0.5 and 3.
    
    verbose : boolean, optional, default is False 
        Print messages
    
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
             
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
            Postfix of the figure filename 
          
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - extent : scalars (left, right, bottom, top), optional, default: None 
            The location, in data-coordinates, of the lower-left and 
            upper-right corners. If `None`, the image is positioned such that 
            the pixel centers fall on zero-based (row, column) indices. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
         
        - format : string, optional, default is 'png' 
            Format to save the figure 
             
        ... and more, see matplotlib    
         
    Returns 
    ------- 
    im_out: smothed or blurred image  
    
    Examples
    --------
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, tcrop=(5,10), fcrop=(0,10000))   
    
    Convert linear spectrogram into dB
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Smooth the spectrogram
    
    >>> Sxx_dB_std05 = maad.sound.smooth(Sxx_dB, std=0.5)
    >>> Sxx_dB_std10 = maad.sound.smooth(Sxx_dB, std=1)
    >>> Sxx_dB_std15 = maad.sound.smooth(Sxx_dB, std=1.5)
    
    Plot spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    >>> maad.util.plot2d(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=10, vmax=70)
    >>> maad.util.plot2d(Sxx_dB_std05, ax=ax2, extent=ext, title='smooth (std=0.5)', vmin=10, vmax=70)
    >>> maad.util.plot2d(Sxx_dB_std10, ax=ax3, extent=ext, title='smooth (std=1)', vmin=10, vmax=70)
    >>> maad.util.plot2d(Sxx_dB_std15, ax=ax4, extent=ext, title='smooth (std=1.5)', vmin=10, vmax=70)
    >>> fig.set_size_inches(7,9)
    >>> fig.tight_layout() 
    
    """ 
    
    if verbose:
        print(72 * '_') 
        print('Smooth the image with a gaussian filter (std = %.1f)' %std) 
     
    # use scikit image (faster than scipy) 
    Sxx_out = filters.gaussian(Sxx,std) 
     
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')   
        cmap   =kwargs.pop('cmap','gray')  
        vmin=kwargs.pop('vmin',np.percentile(Sxx_out,0.1)) 
        vmax=kwargs.pop('vmax',np.percentile(Sxx_out,99.9)) 
        extent=kwargs.pop('extent',None)
            
        if extent is not None : 
            xlabel = 'frequency [Hz]' 
            figsize=kwargs.pop('figsize', (4*2, 0.33*(extent[1]-extent[0])))
        else: 
            xlabel = 'pseudofrequency [points]'
            figsize=kwargs.pop('figsize',(4*2, 13)) 
        
         
        fig, (ax1, ax2) = plt.subplots(2, 1)
        plot2d (Sxx, ax=ax1, extent=extent, figsize=figsize,
                title=('Orignal Spectrogram'),  
                ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                cmap=cmap, **kwargs)
        plot2d (Sxx_out, ax=ax2, extent=extent, figsize=figsize,
                title='Blurred Spectrogram (std='+str(std)+')',  
                ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                cmap=cmap, **kwargs) 
        
        # SAVE FIGURE 
        if savefig is not None :  
            dpi   =kwargs.pop('dpi',96) 
            format=kwargs.pop('format','png') 
            filename=kwargs.pop('filename','_spectro_blurr')              
            filename = savefig+filename+'.'+format 
            print('\n''save figure : %s' %filename) 
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format, 
                        **kwargs)    

    return Sxx_out 
 
