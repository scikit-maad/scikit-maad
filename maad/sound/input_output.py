#!/usr/bin/env python
""" 
Collection of functions to load, read, write, play audio signal or 
its time-frequency representation
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
from scipy.io import wavfile 
from skimage.io import imread 
from scipy.io.wavfile import write as write_wav

# Import internal modules
from maad.util import plot1d, plot2d, linear_scale

#%%
# =============================================================================
# public functions
# =============================================================================
def load(filename, channel='left', detrend=True, verbose=False,
         display=False, savefig=None, **kwargs): 
    """
    Load an audio file (stereo or mono). 
    
    Currently, this function con only load WAVE files.
    
    Parameters
    ----------
    filename : string 
        Name or path of the audio file
    channel : {`'left', right'}, optional, default: left
        In case of stereo sound select the channel that is kept 
    detrend : boolean, optional, default is True
        Subtract the DC value.
    verbose : boolean, optional, default is False
        Print messages into the console or terminal if verbose is True
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
            extent : list of scalars [left, right, bottom, top], optional, default: None
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
    >>> s, fs = maad.sound.load("../data/tropical_forest_morning.wav", channel='left')
    >>> import numpy as np
    >>> tn = np.arange(0,len(s))/fs
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax0, ax1) = plt.subplots(2,1, sharex=True, squeeze=True)
    >>> ax0, _ = maad.util.plot1d(tn,s,ax=ax0, figtitle='ground level')
    >>> ax0.set_ylim((-0.075,0.0751))
    >>> s, fs = maad.sound.load("../data/tropical_forest_morning.wav", channel='right')
    >>> ax1, _ = maad.util.plot1d(tn,s,ax=ax1, figtitle='canopy level')
    >>> ax1.set_ylim((-0.075,0.075))
    >>> fig.tight_layout()
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
        _, fig = plot1d(tn, s_out, figtitle=figtitle, **kwargs)
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

#%%
def load_spectrogram(filename, fs, duration, flims = None, flipud = True,
                verbose=False, display=False, **kwargs): 
    """ 
    Load an image from a file or an URL 
     
    Parameters 
    ----------   
    filename : string 
        Image file name, e.g. ``test.jpg`` or URL. 
     
    fs : scalar 
        Sampling frequency of the audiogram (in Hz) 
     
    duration : scalar 
        Duration of the audiogram (in s) 
        
    flims : list of 2 scalars [min, max], optional, default is None
        flims corresponds to the min and max boundary frequency values
        
    flipud : boolean, optional, default is True 
        Vertical flip of the matrix (image) 
        
    verbose : boolean, optional, default is False
        if True, print message in terminal
     
    display : boolean, optional, default is False 
        if True, display the image 
         
    \*\*kwargs, optional. This parameter is used by plt.plot  
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
    Sxx : ndarray 
        The different color bands/channels are stored in the 
        third dimension, such that a gray-image is MxN, an 
        RGB-image MxNx3 and an RGBA-image MxNx4.    
    tn : 1d ndarray of floats
        time vector (horizontal x-axis)    
    fn : 1d ndarray of floats
        Frequency vector (vertical y-axis)    
    extent : list of scalars [left, right, bottom, top]
        The location, in data-coordinates, of the lower-left and
        upper-right corners. 
        
    Examples
    --------
    >>> xenocanto_link = 'https://www.xeno-canto.org/sounds/uploaded/DTKJSKMKZD/ffts/XC445081-med.png'
    >>> Sxx, tn, fn, ext = maad.sound.load_spectrogram(filename= xenocanto_link,
                                          fs=44100,            
                                          flims=[0,15000],
                                          duration = 10
                                          )
    >>> import matplotlib.pyplot as plt
    >>> maad.util.plot2d(Sxx,extent=ext)
    
    """ 
    
    if verbose :
        print(72 * '_' ) 
        print("loading %s..." %filename)  
     
    # Load image 
    Sxx  = imread(filename, as_gray=True) 
     
    # if 3D, convert into 2D 
    if len(Sxx.shape) == 3: 
        Sxx = Sxx[:,:,0] 
         
    # Rescale the image between 0 to 1 
    Sxx = linear_scale(Sxx, minval= 0.0, maxval=1.0) 
             
    # Get the resolution 
    if flims is None :
        df = fs/(Sxx.shape[0]-1) 
    else:
        df = (flims[1]-flims[0]) / (Sxx.shape[0]-1)
    dt = duration/(Sxx.shape[1]-1) 
    
    # create the vectors
    if flims is None :
        fn = np.arange(0,fs/2,df) 
    else:
        fn = np.arange(flims[0],flims[1],df)  
    tn = np.arange(0,Sxx.shape[1],1) * dt
     
    # Extent 
    extent = [tn[0], tn[-1], fn[0], fn[-1]] 
     
    # flip the image vertically 
    if flipud: Sxx = np.flip(Sxx, 0) 
     
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')  
        title  =kwargs.pop('title','loaded spectrogram') 
        cmap   =kwargs.pop('cmap','gray')  
        figsize=kwargs.pop('figsize',(4, 0.33*(extent[1]-extent[0])))  
        vmin=kwargs.pop('vmin',Sxx.min())  
        vmax=kwargs.pop('vmax',Sxx.max())  
         
        _, fig = plot2d (Sxx, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
     
    return Sxx, tn, fn, extent 
#%%
def write(filename, fs, data):
    """
    Write a NumPy array as a WAV file with the Scipy method. [1]_ 

    Parameters
    ----------
    filename : string or open file handle
        Name of output wav file.
    fs : int
        Sample rate (samples/sec).
    data : ndarray
        Mono or stereo signal as NumPy array.
        
    See Also
    --------
    scipy.io.wavfile.write

    Notes
    -----
    The data-type determines the bits-per-sample and PCM/float.

    Common data types: [2]_

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============

    References
    ----------
    .. [1] The SciPy community, "scipy.io.wavfile.write", v1.6.0.
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
       
    .. [2] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html

    Examples
    --------
    
    Synthesize a 440Hz sine wave at 44100 Hz and write it to disk.
    
    >>> import numpy as np
    >>> fs = 44100; T = 2.0
    >>> t = np.linspace(0, T, int(T*fs))
    >>> data = np.sin(2. * np.pi * 440. *t)
    >>> maad.sound.write('example.wav', fs, data)
    
    Open an audio file, filter a frequency band and write to disk.
    
    >>> from maad import sound
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> s_filt = sound.sinc(s, (3000, 10000), fs)
    >>> sound.write('spinetail_filtered.wav', fs, s_filt)
    """
    if data.ndim > 1:
        data = data.T
    write_wav(filename, fs, np.asfortranarray(data))