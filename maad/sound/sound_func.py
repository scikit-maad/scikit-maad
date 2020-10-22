#!/usr/bin/env python
""" 
Multiresolution Analysis of Acoustic Diversity
functions for processing sound 
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
import scipy as sp 
from scipy.io import wavfile 
from scipy.signal import hilbert,butter, sosfiltfilt, convolve, iirfilter, get_window
from maad.util import plot1D, plot2D, crop_image, linear2dB

# =============================================================================
# private functions
# =============================================================================
def _wave2frames (s, N=512):
    """
    Reshape a sound waveform (ie vector) into a serie of frames (ie matrix) of
    length N
    
    Parameters
    ----------
    s : 1d ndarray of floats (already divided by the number of bits)
        Vector containing the sound waveform 

    N : int, optional, default is 512
        Number of points per frame
                
    Returns
    -------
    timeframes : 2d ndarray of floats
        Matrix containing K frames (row) with N points (column), K*N <= length (s)
    """
    # transform wave into array
    s = np.asarray(s)
    # compute the number of frames
    K = len(s)//N
    # Reshape the waveform (ie vector) into a serie of frames (ie 2D matrix)
    timeframes = s[0:K*N].reshape(-1,N).transpose()
    return timeframes

# =============================================================================
# public functions
# =============================================================================
def load(filename, channel='left', detrend=True, verbose=False,
         display=False, savefig=None, **kwargs): 
    """
    Load a wav file (stereo or mono)
    
    Parameters
    ----------
    filename : string 
        The name or path of the .wav file to load      
        if you want to extract the date of creation of the file, the filename 
        must have this postfix :
        XXXX_yyyymmdd_hhmmss.wav
        with yyyy : year / mm : month / dd: day / hh : hour (24hours) /
        mm : minutes / ss : seconds
            
    channel : {'left', right'}, optional, default: left
        In case of stereo sound select the channel that is kept 

    detrend : boolean, optional, default is True
        Subtract the DC value.
    
    verbose : boolean, optional, default is False
        print messages into the consol or terminal if verbose is True
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
           
        - savefilename : str, optional, default :'_audiogram.png'
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
            ext : list of scalars [left, right, bottom, top], optional, default: None
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
    s_out : 1d ndarray of integer
        Vector containing the audiogram 
        
    fs : int 
        The sampling frequency in Hz of the audiogram    
        
    Examples
    --------
    >>> s, fs = maad.sound.load("guyana_tropical_forest.wav", channel='LEFT', detrend=True, verbose=False)
    >>> import numpy as np
    >>> tn = np.arange(0,len(s))/fs
    >>> maad.util.plot1D(tn,s)
    """
    if verbose :
        print(72 * '_' )
        print("loading %s..." %filename)   
    
    # read the .wav file and return the sampling frequency fs (Hz) 
    # and the audiogram s as a 1D array of integer
    fs, s = wavfile.read(filename)
    if verbose :print("Sampling frequency: %dHz" % fs)
    
    # Normalize the signal between -1 to 1 depending on the type (number of bits)
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
    if detrend: s_out = s_out - np.mean(s_out)
    
    # Time vector
    tn = np.arange(s_out.size)/fs 
    
    # DISPLAY
    if display : 
        figtitle=kwargs.pop('figtitle', 'Orignal sound')        
        _, fig = plot1D(tn, s_out, figtitle=figtitle, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi=kwargs.pop('dpi', 96) 
            bbox_inches=kwargs.pop('bbox_inches', 'tight') 
            format=kwargs.pop('format','png')
            savefilename=kwargs.pop('savefilename', '_audiogram')  
            filename = savefig+savefilename+'.'+format
            if verbose :print('\n''save figure : %s' %filename)
            fig.savefig(fname=filename, dpi=dpi, bbox_inches=bbox_inches,
                        format=format, **kwargs)  
                           
    return s_out, fs

#=============================================================================
def envelope (s, mode='fast', N=32):
    """
    Calcul the envelope of a sound waveform (1d)
    
    Parameters
    ----------
    s : 1d ndarray of floats 
        Vector containing sound waveform
        
    mode : str, optional, default is "fast"
        - "fast" : The sound is first divided into frames (2d) using the 
            function wave2timeframes(s), then the max of each frame gives a 
            good approximation of the envelope.
        - "Hilbert" : estimation of the envelope from the Hilbert transform. 
            The method is slow
    
    N : integer, optional, default is 32
        Size of each frame. The largest, the highest is the approximation.
                  
    Returns
    -------    
    env : 1d ndarray of floats
        Envelope of the sound
        
    References
    ----------
    Fast calculation is inspired by the work of M. Towsey at Queensland University of Technology (QUT).
    
    Examples
    --------
    >>> s,fs = maad.sound.load("guyana_tropical_forest.wav")
    >>> env_fast = maad.sound.envelope(s, mode='fast', N=32)
    >>> env_fast
    array([0.2300415 , 0.28643799, 0.24285889, ..., 0.3059082 , 0.20040894,
       0.26074219])
    
    >>> env_hilbert = maad.sound.envelope(s, mode='hilbert')
    >>> env_hilbert
    array([0.06588196, 0.11301711, 0.09201435, ..., 0.18053983, 0.18351906,
       0.10258595])
    
    compute the time vector for the vector wave
    >>> t = np.arange(0,len(s),1)/fs
    compute the time vector for the vector env_fast
    >>> t_env_fast = np.arange(0,len(env_fast),1)*len(s)/fs/len(env_fast)
    
    plot 0.1 of the envelope and 0.1s of the abs(s)
    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots()
    ax1.plot(t[t<0.1], abs(s[t<0.1]), label='abs(s)')
    ax1.plot(t[t<0.1], env_hilbert[t<0.1], label='env(s) - hilbert option')
    ax1.plot(t_env_fast[t_env_fast<0.1], env_fast[t_env_fast<0.1], label='env(s) - fast option')
    ax1.set_xlabel('Time [sec]')
    ax1.legend()
    """
    if mode == 'fast' :
        # Envelope : take the max (see M. Towsey) of each frame
        frames = _wave2frames(s, N=N)
        env = np.max(abs(frames),0) 
    elif mode =='hilbert' :
        # Compute the hilbert transform of the waveform and take the norm 
        # (magnitude) 
        env = np.abs(hilbert(s))  
    else:
        print ("WARNING : choose a mode between 'fast' and 'hilbert'")
        
    return env

#=============================================================================
def select_bandwidth (x, fs, fcut, forder, fname ='butter', ftype='bandpass', 
                     rp=None, rs=None):
    """
    Lowpass, highpass, bandpass or bandstop a 1d signal with an iir filter
        
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
    
    ftype : {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional, default 
        is 'bandpass'
        The type of filter.
        
    fname : {'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'}, optional, default
        is 'butter'
        
    The type of IIR filter to design:
    
        Butterworth : 'butter'
        
        Chebyshev I : 'cheby1'
        
        Chebyshev II : 'cheby2'
        
        Cauer/elliptic: 'ellip'
        
        Bessel/Thomson: 'bessel'
            
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
    
    Examples
    --------
    
    Load and display the spectrogram of a sound waveform
    
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> p = maad.util.wav2pressure (w, gain=42)
    >>> Pxx,tn,fn,_ = maad.sound.spectrogram(p,fs)
    >>> Lxx = maad.util.power2dBSPL(Pxx) # convert into dB SPL
    >>> fig_kwargs = {'vmax': max(Lxx),
                      'vmin':0,
                      'extent':(tn[0], tn[-1], fn[0], fn[-1]),
                      'figsize':(4,13),
                      'title':'Power spectrogram density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    >>> fig, ax = maad.util.plot2D(Lxx,**fig_kwargs)
    
    Filter the waveform : keep the bandwidth between 6-10kHz
    
    >>> p_filtered = maad.sound.select_bandwidth(p,fs,fcut=[6000,10000], forder=5, fname ='butter', ftype='bandwith')
    >>> Pxx_filtered,tn,fn,_ = maad.sound.spectrogram(p_filtered,fs)
    >>> Lxx_filtered = maad.util.power2dBSPL(Pxx_filtered) # convert into dB SPL
    >>> fig, ax = maad.util.plot2D(Lxx_filtered,**fig_kwargs)
    
    """
    sos = iirfilter(N=forder, Wn=np.asarray(fcut)/(fs/2), btype=ftype,ftype=fname, rp=rp, 
                     rs=rs, output='sos')
    # use sosfiltfilt insteasd of sosfilt to keep the phase of y matches x
    y = sosfiltfilt(sos, x)
    return y
 
#=============================================================================
def fir_filter(x, kernel, axis=0):
    """
    
    1d Finite impulse filter
    => Digital filter based on convolution of 1d kernel over a vector 
    or along an axis of a matrix
    
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
        kernel = ('boxcar', 9)
        kernel = (('gaussian', 0.5), 5)
        kernel = [1 3 5 7 5 3 1] 
        
    axis : int
        Determine along which axis is performed the filtering in case of 2d matrix
        axis = 0 : vertical
        axis = 1 : horizontal
    
    Returns :
    --------
    y : array_like
        The filtered output with the same shape and phase as x
        
    See Also
    --------
    select_bandwidth
    
    Examples
    --------  
    
    Load and display the spectrogram of a sound waveform
    
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> p = maad.util.wav2pressure (w, gain=42)
    >>> Pxx,tn,fn,_ = maad.sound.spectrogram(p,fs)
    >>> Lxx = maad.util.power2dBSPL(Pxx) # convert into dB SPL
    >>> fig_kwargs = {'vmax': max(Lxx),
                      'vmin':0,
                      'extent':(tn[0], tn[-1], fn[0], fn[-1]),
                      'figsize':(4,13),
                      'title':'Power spectrogram density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    >>> fig, ax = maad.util.plot2D(Lxx,**fig_kwargs)
    
    Smooth the waveform (lowpass)
    
    >>> p_filtered = maad.sound.fir_filter(p, kernel=(('gaussian', 2), 5))
    >>> Pxx_filtered,tn,fn,_ = maad.sound.spectrogram(p_filtered,fs)
    >>> Lxx_filtered = maad.util.power2dBSPL(Pxx_filtered) # convert into dB SPL
    >>> fig, ax = maad.util.plot2D(Lxx_filtered,**fig_kwargs)
    
    Smooth the spectrogram, frequency by frequency (blurr)
    >>> Lxx_blurr = maad.sound.fir_filter(Lxx, kernel=(('gaussian', 1), 5), axis=1)
    >>> fig, ax = maad.util.plot2D(Lxx_blurr,**fig_kwargs)
    
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


def spectrogram (x, fs, window='hann', nperseg=1024, noverlap=None, 
                 fcrop=None, tcrop=None, 
                 mode = 'psd',
                 verbose=False, display=False, 
                 savefig = None, **kwargs):
     
    """
    Convert a sound waveform into a spectrogram 
    the output is : 
        - power (mode='psd')
        - amplitude (mode = 'amplitude') => sqrt(power)
        - complex with real and imaginary parts (mode = 'complex')
    
    Parameters
    ----------
    x : 1d ndarray
        Vector containing the sound waveform 
        
    fs : int
        The sampling frequency in Hz 
        
    window : str or tuple or array_like, optional, default to 'hann'
        Desired window to use. If window is a string or tuple, it is passed to 
        get_window to generate the window values, which are DFT-even by default. 
        See get_window for a list of windows and required parameters. 
        If window is array_like it will be used directly as the window and its length must be nperseg. 
        
    nperseg : int, optional. Defaults to 1024.
        Length of the segment used to compute the FFT. No zero padding. 
        For fast calculation, it's better to use a number that is a power 2. 
        This parameter sets the resolution in frequency as the spectrogram will
        contains nperseg/2 frequency bins between 0Hz-(fs/2)Hz, with a resolution
        df = fs/nperseg
        It sets also the time slot (dt) of each frequency frames : dt = nperseg/fs
        The higher is the number, the lower is the resolution in time (dt) 
        but better is the resolution in frequency (df).
        
    noverlap : int, optional. Defaults to None.
        Number of points to overlap between segments. 
        If None, noverlap = nperseg // 2. 

    mode : str, optional. Default is 'psd'
        Choose the output between 
        - 'psd' : Power Spectral Density 
        - 'amplitude' : module of the stft (sqrt(psd))
        - 'complex' : real and imaginary part of the stft
        
    fcrop, tcrop : list of 2 scalars [min, max], optional, default is None
        fcrop corresponds to the min and max boundary frequency values
        tcrop corresponds to the min and max boundary time values  
        
    verbose : boolean, optional, default is False
        print messages into the consol or terminal if verbose is True
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions
            
        - savefilename : str, optional, default :'_filt_audiogram.png'
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
        
        - ext : list of scalars [left, right, bottom, top], optional, default: None
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
    Pxx : 2d ndarray of floats
        Spectrogram : Matrix containing K frames with N/2 frequency bins, 
        K*N <= length (wave)
        
    dt : scalar
        Time resolution of the spectrogram (horizontal x-axis)
        
    df : scalar
        Frequency resolution of the spectrogram (vertical y-axis)
        
    ext : list of scalars [left, right, bottom, top]
        The location, in data-coordinates, of the lower-left and
        upper-right corners. 
    
    Notes
    -----
    This function take care of the energy concervation which is crucial when working in with sound pressure level (dB SPL)
        
    Examples
    --------
    >>> s,fs = maad.sound.load("guyana_tropical_forest.wav")
    
    Compute energy of signal s
    
    >>> E1 = sum(s**2)
    >>> maad.util.linear2dB(E1, mode='power')
    44.861029507805256
    
    Compute the spectrogram with 'psd' output (if N<4096, the energy is lost)
    
    >>> N = 4096
    >>> Pxx,tn,fn,ext = maad.sound.spectrogram (s, fs, nperseg=N, noverlap=N//2, mode = 'psd')   
    
    Display Power Spectrogram
    
    >>> PxxdB = maad.util.linear2dB(Pxx, mode='power') # convert into dB
    >>> fig_kwargs = {'vmax': max(PxxdB),
                      'vmin':-70,
                      'extent':ext,
                      'figsize':(4,13),
                      'title':'Power spectrogram density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    fig, ax = maad.util.plot2D(PxxdB,**fig_kwargs)     
    
    Compute mean power os spectrogram
    
    >>> mean_power = mean(Pxx, axis = 1)
    
    energy => power x time
    
    >>> E2 = sum(mean_power*len(s)) 
    >>> maad.util.linear2dB(E2, mode='power')
    44.93083283875093

    Compute the spectrogram with 'amplitude' output
    
    >>> Sxx,tn,fn,_ = maad.sound.spectrogram (s, fs, nperseg=N, noverlap=N//2, mode='amplitude')  
    
    For energy conservation => convert Sxx (amplitude) into power before doing the average.
    
    >>> mean_power = mean(Sxx**2, axis = 1)
    
    energy => power x time
    
    >>> E3 = sum(mean_power*len(s)) 
    >>>  maad.util.linear2dB(E3, mode='power')
    44.93083283875093
    
    """

    # Test if noverlap is None. By default, noverlap is half the length of the fft
    if noverlap is None :  noverlap = nperseg // 2

    # compute the number of frames
    K = len(x)//(nperseg-noverlap)-1
 
    # compute spectrogram
    fn, tn, Sxx_complex = sp.signal.spectrogram(x, fs, window=window, 
                                              nperseg=nperseg, noverlap=noverlap, 
                                              nfft=nperseg, 
                                              mode='complex',
                                              detrend='constant', 
                                              scaling='density', axis=-1)          
   
            
    if verbose :
        print ('spectrogram dimension Nx=%d Ny=%d' % (Sxx_complex.shape[0], Sxx_complex.shape[1]))
        
    # Mutliply by the frequency resolution step (fs/nperseg) to get the power
    # so multiply by the sqrt((fs/nperseg)) to get the amplitude
    # Also multiply by sqrt(2) in order to compensate that only the postive frequencies are kept
    # complex    
    Sxx_complex = Sxx_complex*np.sqrt(2*(fs/nperseg))
    # magnitude
    Sxx = abs(Sxx_complex)
    # power
    PSDxx = Sxx**2

    if mode =='complex':
        Sxx_out = Sxx_complex
    
    if mode =='amplitude' :
        Sxx_out = Sxx
    
    if mode =='psd':
        Sxx_out = PSDxx    
    
    # test if the last frames are computed on a whole time frame. 
    # if note => remove these frames
    if PSDxx.shape[1] > K:
        sup = Sxx_out.shape[1] - K
        Sxx_out = Sxx_out[:,:-sup]
        tn = tn[:-sup]
        
    # Remove the last frequency bin in order to obtain nperseg/2 frequency bins
    # instead of nperseg/2 + 1 
    Sxx_out = Sxx_out[:-1,:]
    fn = fn[:-1]
    
    # Crop the image in order to analyzed only a portion of it
    if (fcrop or tcrop) is not None:
        if verbose:print ('Crop the spectrogram along time axis and frequency axis')
        Sxx_out, tn, fn = crop_image(Sxx_out,tn,fn,fcrop,tcrop)

    if verbose:
        print('max value of the spectrogram %.5f' % Sxx_out.max())

    # Extent
    ext = [tn[0], tn[-1], fn[0], fn[-1]]
    # dt and df resolution
    dt = tn[1]-tn[0]
    df = fn[1]-fn[0]

    if verbose==True:
        print("*************************************************************")
        print("   Time resolution dt=%.2fs | Frequency resolution df=%.2fHz "
              % (dt, df))  
        print("*************************************************************")
    
    # Display
    if display : 
        
        #### convert into dB if db_range is not none
        db_range =kwargs.pop('db_range',90)
        if db_range is not None : 
            if mode=='psd':  
                Sxx_disp = linear2dB(Sxx_out, db_range=db_range, mode='power')
            if mode=='amplitude': 
                Sxx_disp = linear2dB(Sxx_out, db_range=db_range, mode='amplitude')
            if mode=='complex': 
                Sxx_disp = linear2dB(abs(Sxx_out), db_range=db_range, mode='amplitude') 
        else:
            Sxx_disp = Sxx_out

        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Spectrogram')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',Sxx_disp.min()) 
        vmax=kwargs.pop('vmax',Sxx_disp.max()) 
        
        _, fig = plot2D (Sxx_disp, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax,
                         cmap=cmap, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            bbox_inches=kwargs.pop('bbox_inches', 'tight') 
            format=kwargs.pop('format','png')
            savefilename=kwargs.pop('savefilename', '_spectrogram')  
            filename = savefig+savefilename+'.'+format
            print('\n''save figure : %s' %filename)
            fig.savefig(fname=filename, dpi=dpi, bbox_inches=bbox_inches,
                        format=format, **kwargs)     

    return Sxx_out, tn, fn, ext   







