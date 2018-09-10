#!/usr/bin/env python
""" functions for processing sound """
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
from scipy.signal import butter, sosfilt, hann, stft
from ..util import plot1D, plot2D, db_scale, crop_image, date_from_filename

def load(filename, channel='left', display=False, savefig=None, **kwargs): 
    """
    Load a wav file (stereo or mono)
    
    Parameters
    ----------
    filename : string 
        The name or path of the .wav file to load
            
    channel : {'left', right'}, optional, default: left
        In case of stereo sound select the channel that is kept 
        
    Returns
    -------    
    s_out : 1d ndarray of integer
        Vector containing the audiogram 
        
    fs : int 
        The sampling frequency in Hz of the audiogram  
    
    tn : 1d ndarray of scalar
        Vector containing the sample time
       
    """
    
    print("loading wav file...")
    
    # read the .wav file and return the sampling frequency fs (Hz) 
    # and the audiogram s as a 1D array of integer
    fs, s = wavfile.read(filename)
    print("Sampling frequency: %dHz" % fs)
    
    # Be sure that 's' is an array of integer 16bits
    s.astype('int16')
    
    # Normalize the signal between -1 to 1 by dividing with 2**15
    s = s/2**15
    
    # test if stereo signal. if YES => keep only the ch_select
    if s.ndim==2 :
        if channel == 'left' :
            print("Select left channel")
            s_out = s[:,0] - np.mean(s[:,0]) 
        else:
            print("Select right channel")
            s_out = s[:,1] - np.mean(s[:,1]) 
    else:
        s_out = s;
        
    # Detrend the signal by removing the DC offset
    s_out = s_out - np.mean(s_out)
    
    # Time vector
    tn = np.arange(s_out.size)/fs 
    
    # get the date from the filename
    date = date_from_filename (filename)
    
    # DISPLAY
    if display : 
        _, fig = plot1D(tn, s_out, figtitle = 'Orignal sound')
        # SAVE FIGURE
        if savefig is not None : 
            filename = savefig+'_audiogram.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')       
            
    return s_out, fs, date

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
    
    order : int, optional, default: 5
        Order of the Butterworth filter. 
                      
    Returns
    -------
    s_out : 1d ndarray of integer
        Vector containing the audiogram after being filtered           
    """    

    if lfc is None:
        lfc = 0
    if hfc is None:
        hfc = fs/2         
    
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
                          xlabel=xlabel, ylabel=ylabel, **kwargs) 
        # plot filtered sound
        ax2, fig = plot1D(tn, s_out, ax = ax1, figtitle = figtitle, linecolor = linecolor,
               legend='filtered sound',xlabel=xlabel, ylabel=ylabel,**kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png')
            filename=kwargs.pop('filename','_filt_audiogram')             
            filename = savefig+filename+'.'+format
            
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs)   

    return s_out


def convert_dt_df_into_points(dt, df, fs):
    
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

def spectrogram(s, fs, nperseg=512, overlap=0.5, db_range=60, db_gain=20, db_norm_val=None, 
                rescale=None, fcrop=None, tcrop=None, display=False, savefig = None, **kwargs):
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

    Returns
    -------
    fn : 1d ndarray of scalar
        Array of sample frequencies.
            
    tn : 1d ndarray of scalar
        Array of segment times.
            
    Sxx : ndarray
        Spectrogram of s equivalent to Matlab       
    """   
    
    noverlap = round(overlap*nperseg) 

    # sliding window 
    win = hann(nperseg)
    
    print("Computing spectrogram with nperseg=%d and noverlap=%d..." % (nperseg, noverlap))
    
    # spectrogram function from scipy via stft
    # Normalize by win.sum()
    fn, tn, Sxx = stft(s, fs, win, nperseg, 
                       noverlap, nfft=nperseg, detrend=False, 
                       return_onesided=True, boundary=None, 
                       padded=True, axis=-1)
    
    print(fn[0])
    print(fn[-1])
    print(tn[0])
    print(tn[-1])
    

    # stft (complex) without normalisation
    scale_stft = sum(win)/len(win)
    Sxx = Sxx / scale_stft  # remove normalization
    Sxx = np.abs(Sxx)       # same result as abs(spectrogram) with Matlab
    
    print('max %.5f' % Sxx.max())
 
    # Convert in dB scale 
    if db_range is not None :
        print ('Convert in dB scale')
        Sxx = db_scale(Sxx, db_range,db_gain,db_norm_val)
        
    # Rescale
    if rescale is not None :
        print ('Linear rescale between 0 to 1 (if db_norm_val is None)')
        Sxx = (Sxx + db_range)/db_range
    
    # Crop the image in order to analyzed only a portion of it
    if (fcrop or tcrop) is not None:
        print ('Crop the spectrogram along time axis and frequency axis')
        Sxx, tn, fn = crop_image(Sxx,tn,fn,fcrop,tcrop)
    
    # Extent
    ext = [tn[0], tn[-1], fn[0], fn[-1]]
    
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Spectrogram')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        _, fig = plot2D (Sxx, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax,
                         cmap=cmap, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png')
            filename=kwargs.pop('filename','_spectrogram')             
            filename = savefig+filename+'.'+format
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs)   
    
    print("Time resolution dt=%.2fs // Frequency resolution df=%.2fHz" 
          % (tn[1]-tn[0], fn[1]-fn[0]))
 
    return Sxx, tn , fn, ext


def preprocess_wrapper(filename, display=False, savefig=None, **kwargs):

     
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
    db_norm_val=kwargs.pop('db_norm_val', 1)
    rescale=kwargs.pop('rescale', True)    
    fcrop=kwargs.pop('fcrop', None)
    tcrop=kwargs.pop('tcrop', None)    
    dt=kwargs.pop('dt', 0.02)
    df=kwargs.pop('df', 20)  
    overlap=kwargs.pop('overlap', None)
    nperseg=kwargs.pop('nperseg', None)
    if overlap is None:
        overlap,_,dt,df = convert_dt_df_into_points(dt=0.02,df=20,fs=fs)
    elif (dt or df) is not None :
        print ("Warning dt and df are not taken into account. Priority to overlap")    
    if nperseg is None:
        _,nperseg,dt,df = convert_dt_df_into_points(dt=0.02,df=20,fs=fs) 
    elif (dt or df) is not None :
        print ("Warning dt and df are not taken into account. Priority to nperseg")   
            
    """=======================================================================
    # Compute the spectrogram of the sound and convert into dB
    ======================================================================="""
    Sxx,tn,fn,ext = spectrogram(s_filt, fs, nperseg=nperseg, overlap=overlap, 
                                db_range=db_range, db_gain=db_gain, 
                                db_norm_val= db_norm_val, rescale=rescale, 
                                fcrop=fcrop, tcrop=tcrop, display=display, 
                                savefig=savefig, **kwargs)
    
    dt = tn[1] - tn[0]
    df = fn[1] - fn[0]

    print('lfc:%d df:%.2fHz dt:%.2fs' % (lfc, df, dt)) 
   
    return Sxx, fs, ext, date






