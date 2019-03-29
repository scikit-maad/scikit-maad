#!/usr/bin/env python
""" Multiresolution Analysis of Acoustic Diversity
    functions for processing sound """
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
from scipy.signal import butter, sosfilt, hann, stft
from ..util import plot1D, plot2D, db_scale, crop_image, date_from_filename, linear_scale



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
        ****************************************************    
        savefilename : str, optional, default :'_audiogram.png'
            Postfix of the figure filename
        **************************************************** 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : list of scalars [left, right, bottom, top], optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure
            
        ... and more, see matplotlib     
        
    Returns
    -------    
    s_out : 1d ndarray of integer
        Vector containing the audiogram 
        
    fs : int 
        The sampling frequency in Hz of the audiogram         
    """
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

def select_bandwidth(s,fs, lfc=None, hfc=None, order=3, display=False, 
                     savefig=None, **kwargs):
    """
    select a bandwidth of the signal
    
    Parameters
    ----------
    s :  1d ndarray of integer
        Vector containing the audiogram     
        
    fs : int
        The sampling frequency in Hz
        
    lfc : int, optional, default: None
        Low frequency cut (Hz) in the range [0;fs/2]   
        
    hfc : int, optional, default: None
        High frequency cut (Hz) in the range [0;fs/2]
        if lfc and hfc are declared and lfc<hfc, bandpass filter is performed
        
    order : int, optional, default: 3
        Order of the Butterworth filter. 
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        ****************************************************    
        savefilename : str, optional, default :'_filt_audiogram.png'
            Postfix of the figure filename
        **************************************************** 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : list of scalars [left, right, bottom, top], optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure
        
        ... and more, see matplotlib     
          
    Returns
    -------
    s_out : 1d ndarray of integer
        Vector containing the audiogram after being filtered           
    """    


    if lfc is None:
        lfc = 0
    if hfc is None:
        hfc = fs/2         
    
    print(72 * '_')
    if lfc!=0 and hfc!=fs/2:
        # bandpass filter the signal
        Wn = [lfc/fs*2, hfc/fs*2]
        print("Bandpass filter [%dHz; %dfHz] in progress..." % (lfc, hfc))
        sos = butter(order, Wn, analog=False, btype='bandpass', output='sos')
        s_out = sosfilt(sos, s) 
    elif lfc==0 and hfc!=fs/2 :     
        # lowpass filter the signal
        Wn = hfc/fs*2
        print("Lowpass filter <%dHz in progress..." % hfc)
        sos = butter(order, Wn, analog=False, btype='lowpass', output='sos')
        s_out = sosfilt(sos, s)    
    elif lfc!=0 and hfc==fs/2 :  
        # highpass filter the signal
        Wn = lfc/fs*2
        print("Highpass filter >%dHz in progress..." % lfc)
        sos = butter(order, Wn, analog=False, btype='highpass', output='sos')
        s_out = sosfilt(sos, s)       
    else:
        # do nothing
        print("No filtering in progress")
        s_out = s
        
    if display : 
        figtitle =kwargs.pop('figtitle','Audiogram')
        ylabel =kwargs.pop('ylabel','Amplitude [AU]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        figtitle  =kwargs.pop('figtitle','Audiogram')
        linecolor  =kwargs.pop('linecolor','r')
        # Time vector
        tn = np.arange(s.size)/fs   
        # plot Original sound
        ax1, fig = plot1D(tn, s, figtitle = figtitle, linecolor = 'k' , legend='orignal sound',
                          xlabel=xlabel, ylabel=ylabel, now=False, **kwargs) 
        # plot filtered sound
        ax2, fig = plot1D(tn, s_out, ax = ax1, figtitle = figtitle, linecolor = linecolor,
               legend='filtered sound',xlabel=xlabel, ylabel=ylabel, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi=kwargs.pop('dpi', 96) 
            bbox_inches=kwargs.pop('bbox_inches', 'tight') 
            format=kwargs.pop('format','png')
            savefilename=kwargs.pop('savefilename', '_filt_audiogram')  
            filename = savefig+savefilename+'.'+format
            print('\n''save figure : %s' %filename)
            fig.savefig(fname=filename, dpi=dpi, bbox_inches=bbox_inches,
                        format=format, **kwargs)  
    
    return s_out

def _convert_dt_df_into_points(dt, df, fs):
    
    nperseg = round(fs/df)
    # find the closest power of 2 
    if 2**(nperseg-1).bit_length() - nperseg < nperseg- 2**(nperseg-1).bit_length()/2 :
        nperseg = 2**(nperseg-1).bit_length()
    else :
        nperseg = 2**(nperseg-1).bit_length()/2

    overlap = 1-round(dt*fs)/nperseg
    
    df = fs/nperseg
    dt = (1-overlap)*nperseg/fs
    return overlap,int(nperseg), dt, df


def spectrogram(s, fs, nperseg=512, overlap=0.5, dt_df_res=None, db_range=None, db_gain=20,  
                rescale=False, fcrop=None, tcrop=None, display=False, 
                savefig = None, **kwargs):
    """
    Calcul the spectrogram of a signal s
    
    Parameters
    ----------
    s : 1d ndarray of integer
        Vector containing the audiogram   
        
    fs : int
        The sampling frequency in Hz       
        
    nperseg : int, optional, default: 512
        Number of points par segment (short window).
        This parameter sets the resolution in time of the spectrogram,
        the higher is the number, the lower is the resolution in time but
        better is the resolution in frequency.
        This number is used to compute the short fourier transform (sfft).
        For fast calculation, it's better to use a number that is a power 2.  
        
    overlap : scalar, optional, default: 0.5
        Pourcentage of overlap between each short windows
        The number ranges between 0 (no overlap) and 1 (complete overlap)
        
    dt_df_res : list of two scalars [dt, df], optional, default is None
        **Priority to dt_df_res is provided**, 
        nperseg and overlap are not taken into account
        usage : 
            dt_df_res = [0.02, 20] means
            time resolution dt = 0.02s / frequency resolution df = 20Hz
            
    db_range : int, optional, default is None
        Final dB range of the spectrogram values.
        If dB_range is None, no db scale is performed. Output is linear
        
    db_gain : int, optional, default is 20
        After db scale, a db gain is added to the spectrogram values.
        
    rescale : boolean, optional, default is False
        a linear rescale is performed between 0 to 1 on the final (linear 
        or dB scale) spectrogram.
        The spectrogram can be in dB scale or linear scale
        
    fcrop, tcrop : list of 2 scalars [min, max], optional, default is None
        fcrop corresponds to the min and max boundary frequency values
        tcrop corresponds to the min and max boundary time values    
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        ****************************************************    
        savefilename : str, optional, default :'_filt_audiogram.png'
            Postfix of the figure filename
        **************************************************** 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : list of scalars [left, right, bottom, top], optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure
        
        ... and more, see matplotlib  
  
    Returns
    -------        
    Sxx : ndarray
        Spectrogram (collection of Power Spectrum Density (PSD)) of s.
        Spectrogram(s) = abs(STFD)²
        This is equivalent to Audacity, in linear or dB scale. Values
        is normalized between 0 to 1 if rescale is True
        
    dt : scalar
        Time resolution of the spectrogram (horizontal x-axis)
        
    df : scalar
        Frequency resolution of the spectrogram (vertical y-axis)
        
    ext : list of scalars [left, right, bottom, top]
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.
        
    date : object datetime
        This object contains the date of creation of the file extracted from
        the filename postfix. 
        The filename must follow this format :
            XXXX_yyyymmdd_hhmmss.wav
            with yyyy : year / mm : month / dd: day / hh : hour (24hours) /
            mm : minutes / ss : seconds
    """   

    # Convert dt and df into overlap and nperseg (pixel-based)
    if dt_df_res is not None :
        dt, df = dt_df_res
        overlap,nperseg,_,_ = _convert_dt_df_into_points(dt, df, fs)
    
    noverlap = round(overlap*nperseg) 

    # sliding window 
    win = hann(nperseg)
    
    print(72 * '_')
    print("Computing spectrogram with nperseg=%d and noverlap=%d..." % (nperseg, noverlap))
    
    # spectrogram function from scipy via stft
    # Normalize by win.sum()
    fn, tn, Sxx = stft(s, fs, win, nperseg, noverlap, nfft=nperseg)
    
    # stft (complex) without normalisation
    scale_stft = sum(win)/len(win)
    Sxx = Sxx / scale_stft      # normalization 
    Sxx = np.abs(Sxx)**2        # Get the PSD (power spectra density)
    
    print('max value in the audiogram %.5f' % Sxx.max())
 
    # Convert in dB scale 
    if db_range is not None :
        print ('Convert in dB scale')
        Sxx = db_scale(Sxx, db_range, db_gain)
        vmin = -db_range
        vmax = 0
    else:
    # stay in linear scale
        vmin = np.min(Sxx)
        vmax = np.max(Sxx)              

    # Crop the image in order to analyzed only a portion of it
    if (fcrop or tcrop) is not None:
        print ('Crop the spectrogram along time axis and frequency axis')
        Sxx, tn, fn = crop_image(Sxx,tn,fn,fcrop,tcrop)
    
    # Extent
    ext = [tn[0], tn[-1], fn[0], fn[-1]]
    # dt and df resolution
    dt = tn[1]-tn[0]
    df = fn[1]-fn[0]
    print("*************************************************************")
    print("   Time resolution dt=%.2fs | Frequency resolution df=%.2fHz "
          % (dt, df))  
    print("*************************************************************")
           
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Spectrogram')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',vmin) 
        vmax=kwargs.pop('vmax',vmax) 
        _, fig = plot2D (Sxx, extent=ext, figsize=figsize,title=title, 
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

    # Rescale
    if rescale :
        print ('Linear rescale between 0 to 1')
        if db_range is not None : 
            Sxx = (Sxx + db_range)/db_range
        else:
            Sxx = linear_scale(Sxx, minval= 0.0, maxval=1.0)
                
  
    return Sxx, dt, df, ext


def spectrogram2(s, fs, nperseg=512, overlap=0, dt_df_res=None, detrend=False,
                mode='amplitude', fcrop=None, tcrop=None, 
                verbose=False, display=False, 
                savefig = None, **kwargs):
    """
    Calcul the spectrogram of a signal s
    
    Parameters
    ----------
    s : 1d ndarray of integer
        Vector containing the audiogram   
        
    fs : int
        The sampling frequency in Hz       
        
    nperseg : int, optional, default: 512
        Number of points par segment (short window).
        This parameter sets the resolution in time of the spectrogram,
        the higher is the number, the lower is the resolution in time but
        better is the resolution in frequency.
        This number is used to compute the short fourier transform (sfft).
        For fast calculation, it's better to use a number that is a power 2.  
        
    overlap : scalar, optional, default: 0.5
        Pourcentage of overlap between each short windows
        The number ranges between 0 (no overlap) and 1 (complete overlap)
        
    dt_df_res : list of two scalars [dt, df], optional, default is None
        **Priority to dt_df_res is provided**, 
        nperseg and overlap are not taken into account
        usage : 
            dt_df_res = [0.02, 20] means
            time resolution dt = 0.02s / frequency resolution df = 20Hz
            
    detrend : boolean, optional, default is False
        if detrend is True, the DC value (ie. mean value of signal) is
        subtracted from the signal. This results by a lower value at 0Hz
            
    mode : str, optional, default is 'amplitude'
        select the output values of spectrogram :
            - 'amplitude' : Sxx = A
            - 'psd'       : Sxx = A² (= Power Spectrum Density (PSD))
        
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
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        ****************************************************    
        savefilename : str, optional, default :'_filt_audiogram.png'
            Postfix of the figure filename
        **************************************************** 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : list of scalars [left, right, bottom, top], optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure
        
        ... and more, see matplotlib  
  
    Returns
    -------        
    Sxx : ndarray
        Spectrogram (collection of Power Spectrum Density (PSD)) of s.
        Spectrogram(s) = abs(STFD)²
        This is equivalent to Audacity, in linear or dB scale. Values
        is normalized between 0 to 1 if rescale is True
        
    dt : scalar
        Time resolution of the spectrogram (horizontal x-axis)
        
    df : scalar
        Frequency resolution of the spectrogram (vertical y-axis)
        
    ext : list of scalars [left, right, bottom, top]
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.
        
    date : object datetime
        This object contains the date of creation of the file extracted from
        the filename postfix. 
        The filename must follow this format :
            XXXX_yyyymmdd_hhmmss.wav
            with yyyy : year / mm : month / dd: day / hh : hour (24hours) /
            mm : minutes / ss : seconds
    """   
    # Convert dt and df into overlap and nperseg (pixel-based)
    if dt_df_res is not None :
        dt, df = dt_df_res
        overlap,nperseg,_,_ = _convert_dt_df_into_points(dt, df, fs)
    
    noverlap = round(overlap*nperseg) 
    
    # transform s into array
    s = np.asarray(s)
    
   # compute the number of frames
    K = len(s)//nperseg

    # sliding window 
    #win = tukey(nperseg, 1/32)
    #win = np.ones(nperseg)
    win = hann(nperseg)
    
    if verbose:
        print(72 * '_')
        print("Computing spectrogram with nperseg=%d and noverlap=%d..." 
              % (nperseg, noverlap))
    
    if mode == 'amplitude':
        fn, tn, Sxx = sp.signal.spectrogram(s, fs, win, nperseg=nperseg, 
                                            noverlap=noverlap, nfft=nperseg, 
                                            scaling ='spectrum', mode='complex', 
                                            detrend=detrend)    
    if mode == 'psd':
        fn, tn, Sxx = sp.signal.spectrogram(s, fs, win, nperseg=nperseg, 
                                            noverlap=noverlap, nfft=nperseg, 
                                            scaling ='spectrum', mode='psd', 
                                            detrend=detrend)    
        
    # Get the magnitude of the complex and multiply by 2 (because take only 
    # half of the spectrum (positive frequencies))
    Sxx = abs(Sxx)*2
   
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

    # Crop the image in order to analyzed only a portion of it
    if (fcrop or tcrop) is not None:
        if verbose:print ('Crop the spectrogram along time axis and frequency axis')
        Sxx, tn, fn = crop_image(Sxx,tn,fn,fcrop,tcrop)
    
    # Extent
    ext = [tn[0], tn[-1], fn[0], fn[-1]]
    # dt and df resolution
    dt = tn[1]-tn[0]
    df = fn[1]-fn[0]
    
    if verbose:
        print("*************************************************************")
        print("   Time resolution dt=%.2fs | Frequency resolution df=%.2fHz "
              % (dt, df))  
        print("   Maximum value of the spectrogram : df=%.2f" % (abs(np.max(Sxx))))
        print("*************************************************************")
           
    # Display
    if display :        
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Spectrogram')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',-120) 
        vmax=kwargs.pop('vmax',0) 
     
        # transform data in dB
        if mode == 'psd':
            Sxx = np.sqrt(Sxx)
        #### convert into dB
        SxxdB = db_scale(Sxx, db_range=-vmin, db_gain=0)
        
        # Plot
        _, fig = plot2D (SxxdB, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax,
                         cmap=cmap, **kwargs)
        
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            bbox_inches=kwargs.pop('bbox_inches', 'tight') 
            format=kwargs.pop('format','png')
            savefilename=kwargs.pop('savefilename', '_spectrogram')  
            filename = savefig+savefilename+'.'+format
            if verbose:print('\n''save figure : %s' %filename)
            fig.savefig(fname=filename, dpi=dpi, bbox_inches=bbox_inches,
                        format=format, **kwargs)                       
            
    return Sxx, tn, fn, ext


def preprocess_wrapper(filename, display=False, savefig=None, **kwargs):
    """
    Wrapper function to preprocess an audio recorder into a spectrogram
    
    Parameters
    ----------
    filename : string 
        The name or path of the .wav file to load      
        if you want to extract the date of creation of the file, the filename 
        must have this postfix :
            XXXX_yyyymmdd_hhmmss.wav
            with yyyy : year / mm : month / dd: day / hh : hour (24hours) /
            mm : minutes / ss : seconds    
            
    display : boolean, optional, default is False
        Display the signals and the spectrograms if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by the maad function as well
        as the plt.plot and savefig functions.
        All the input arguments required or optional in the signature functions
        can be passed.
        see the signature of each maad function to know the parameters 
        that can be passed as kwargs :
            load(filename, channel='left', display=False, savefig=None, 
                 **kwargs)
            select_bandwidth(s,fs, lfc=None, hfc=None, order=3, display=False, 
                             savefig=None, **kwargs):
            spectrogram(s, fs, nperseg=512, overlap=0.5, dt_df_res=None, 
                        db_range=60, db_gain=20, rescale=True, fcrop=None, 
                        tcrop=None, display=False, savefig = None, **kwargs):
        ... and more, see matplotlib  
        
        example : 
            preprocess_wrapper('audio.wav', display=False, savefig=None, 
                               tcrop=[0,30])
  
    Returns
    ------- 
    Sxx : 2d ndarray
        Spectrogram of s equivalent to Audacity     
        
    fs : Scalar
        Sampling frequency
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.   
    
    date : object datetime
        This object contains the date of creation of the file extracted from
        the filename postfix. 
        The filename must follow this format :
            XXXX_yyyymmdd_hhmmss.wav
            with yyyy : year / mm : month / dd: day / hh : hour (24hours) /
            mm : minutes / ss : seconds   
    """ 

    channel=kwargs.pop('channel','left')
    """========================================================================
    # Load the original sound
    ======================================================================="""
    s,fs,date = load(filename=filename, channel=channel, display=display, 
                             savefig=savefig, **kwargs)
    
    lfc=kwargs.pop('lfc', 500) 
    hfc=kwargs.pop('hfc', None) 
    order=kwargs.pop('order', 3)    
    """=======================================================================
    # Filter the sound between Low frequency cut (lfc) and High frequency cut (hlc)
    ======================================================================="""
    s_filt = select_bandwidth(s, fs, lfc=lfc, hfc=hfc, order=order, 
                              display=display, savefig=savefig, **kwargs)
    
    db_range=kwargs.pop('db_range', 60)
    db_gain=kwargs.pop('db_gain', 30)
    rescale=kwargs.pop('rescale', True)    
    fcrop=kwargs.pop('fcrop', None)
    tcrop=kwargs.pop('tcrop', None)    
    dt,df=kwargs.pop('dt_df_res', [0.02, 20])
    overlap=kwargs.pop('overlap', None)
    nperseg=kwargs.pop('nperseg', None)
    if overlap is None:
        overlap,_,dt,df =_convert_dt_df_into_points(dt=dt,df=df,fs=fs)
    elif (dt or df) is not None :
        print ("Warning dt and df are not taken into account. Priority to overlap")    
    if nperseg is None:
        _,nperseg,dt,df =_convert_dt_df_into_points(dt=dt,df=df,fs=fs) 
    elif (dt or df) is not None :
        print ("Warning dt and df are not taken into account. Priority to nperseg")   
            
    """=======================================================================
    # Compute the spectrogram of the sound and convert into dB
    ======================================================================="""
    Sxx,dt,df,ext = spectrogram(s_filt, fs, nperseg=nperseg, overlap=overlap, 
                                db_range=db_range, db_gain=db_gain,rescale=rescale,  
                                fcrop=fcrop, tcrop=tcrop, display=display, 
                                savefig=savefig, **kwargs)
   
    return Sxx, fs, dt, df, ext, date






