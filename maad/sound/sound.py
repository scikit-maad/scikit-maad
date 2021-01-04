#!/usr/bin/env python
""" 
Collection of functions to load and preprocess audio signals
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
from scipy.signal import hilbert, sosfiltfilt, convolve, iirfilter, get_window
from maad.util import (plot1D, plot2D, crop_image, power2dB, amplitude2dB, 
                       mean_dB, get_unimode, running_mean) 
from maad.rois import remove_background_along_axis

# =============================================================================
# private functions
# =============================================================================
def wave2frames (s, Nt=512):
    """
    Reshape a sound waveform (ie vector) into a serie of frames (ie matrix) of
    length Nt
    
    Parameters
    ----------
    s : 1d ndarray of floats (already divided by the number of bits)
        Vector containing the sound waveform 
    Nt : int, optional, default is 512
        Number of points per frame
                
    Returns
    -------
    timeframes : 2d ndarray of floats
        Matrix containing K frames (row) with Nt points (column), K*N <= length (s)
    """
    # transform wave into array
    s = np.asarray(s)
    # compute the number of frames
    K = len(s)//Nt
    # Reshape the waveform (ie vector) into a serie of frames (ie 2D matrix)
    timeframes = s[0:K*Nt].reshape(-1,Nt).transpose()
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
    channel : {`'left', right'}, optional, default: left
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
def envelope (s, mode='fast', Nt=32):
    """
    Calcul the envelope of a sound waveform (1d)
    
    Parameters
    ----------
    s : 1d ndarray of floats 
        Vector containing sound waveform 
    mode : str, optional, default is `fast`
        - `fast` : The sound is first divided into frames (2d) using the 
            function wave2timeframes(s), then the max of each frame gives a 
            good approximation of the envelope.
        - `Hilbert` : estimation of the envelope from the Hilbert transform. 
            The method is slow
    Nt : integer, optional, default is `32`
        Size of each frame. The largest, the highest is the approximation.
                  
    Returns
    -------    
    env : 1d ndarray of floats
        Envelope of the sound
        
    References
    ----------
    ..[1] Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms 
          Derived from Natural Recordings of the Environment. 
          Queensland University of Technology, Brisbane.
    ..[2] Towsey, Michael (2017),The calculation of acoustic indices derived 
          from long-duration recordings of the naturalenvironment.
          Queensland University of Technology, Brisbane.
    
    Examples
    --------
    >>> s,fs = maad.sound.load("guyana_tropical_forest.wav")
    >>> env_fast = maad.sound.envelope(s, mode='fast', Nt=32)
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
    
    plot 0.1s of the envelope and 0.1s of the abs(s)
    
    >>> import matplotlib.pyplot as plt
    >>> fig1, ax1 = plt.subplots()
    >>> ax1.plot(t[t<0.1], abs(s[t<0.1]), label='abs(s)')
    >>> ax1.plot(t[t<0.1], env_hilbert[t<0.1], label='env(s) - hilbert option')
    >>> ax1.plot(t_env_fast[t_env_fast<0.1], env_fast[t_env_fast<0.1], label='env(s) - fast option')
    >>> ax1.set_xlabel('Time [sec]')
    >>> ax1.legend()
    """
    if mode == 'fast' :
        # Envelope : take the max (see M. Towsey) of each frame
        frames = wave2frames(s, Nt)
        env = np.max(abs(frames),0) 
    elif mode =='hilbert' :
        # Compute the hilbert transform of the waveform and take the norm 
        # (magnitude) 
        env = np.abs(hilbert(s))  
    else:
        print ("WARNING : choose a mode between 'fast' and 'hilbert'")
        
    return env

#=============================================================================
def intoOctave (X, fn, thirdOctave=True, display=False, **kwargs):
    """
    Transform a linear spectrum (1d) or Spectrogram (2d into octave or 1/3 octave
    spectrum (1d) or Spectrogram (2d)
    Better to work with PSD (amplitude²) for energy conservation

    Parameters
    ----------
    X : ndarray of floats
        Linear spectrum (1d) or Spectrogram (2d). 
        Work with PSD to be consistent with energy concervation
    fn : 1d ndarray of floats
        Frequency vector of the linear spectrum/spectrogram
    thirdOctave : Boolean, default is True
        choose between Octave or thirdOctave frequency resolution
    display : boolean, default is False
        Display the octave spectrum/spectrogram
    ** kwargs : optional. This parameter is used by plt.plot    
    
    Returns
    -------
    X_octave : ndarray of floats
        Octave or 1/3 octave Spectrum (1d) or Spectrogram (2d)
    bin_octave : vector of floats
        New frequency vector (octave or 1/3 octave frequency repartition)
    """

    # define the third octave or octave frequency vector in Hz.
    if thirdOctave :
        bin_octave = np.array([16,20,25,31.5,40,50,63,80,100,125,160,200,250,315,
                               400,500,630,800,1000,1250,1600,2000,2500,3150,4000,
                               5000,6300,8000,10000,12500,16000,20000]) # third octave band.
    else:
        bin_octave = np.array([16,31.5,63,125,250,500,1000,2000,4000,8000,16000]) # octave

    
    # get the corresponding octave from fn
    bin_octave = bin_octave[(bin_octave>=np.min(fn)) & (bin_octave<=np.max(fn))]
    
    # Bins limit
    bin_octave_low = bin_octave/(2**0.1666666)
    bin_octave_up = bin_octave*(2**0.1666666)
       
    # select the indices corresponding to the frequency bins range
    X_octave = []
    for ii in np.arange(len(bin_octave)):
        ind = (fn>=bin_octave_low[ii])  & (fn<=bin_octave_up[ii])
        X_octave.append(np.sum(X[ind,], axis=0))
    
    X_octave = np.asarray(X_octave)
            
    if display :
        X_octave_dB = power2dB(X_octave)
        if np.ndim(X_octave_dB) == 2 :
            extent = kwargs.pop('extent',None)
            if extent is not None : 
                xlabel = 'Time [sec]'
                figsize = (4, 0.33*(extent[1]-extent[0]))
            else: 
                xlabel = 'pseudoTime [points]'
                figsize = (4,13)
            
            fig_kwargs = {'vmax': kwargs.pop('vmax',np.max(X_octave_dB)),
                          'vmin': kwargs.pop('vmin',np.min(X_octave_dB)),
                          'extent':extent,
                          'figsize':kwargs.pop('figsize',figsize),
                          'yticks' : (np.arange(len(bin_octave)), bin_octave),
                          'title':'Octave Spectrogram',
                          'xlabel': xlabel,
                          'ylabel':'Frequency [Hz]',
                          }
            plot2D(X_octave_dB,**fig_kwargs)
        elif np.ndim(X_octave_dB) == 1 :
            
            fig_kwargs = {
                          'title':'Octave Spectrum',
                          'xlabel': kwargs.pop('xlabel','Frequency [Hz]'),
                          'ylabel': kwargs.pop('ylabel','Amplitude [dB]'),
                          }
            plot1D(bin_octave, X_octave_dB,**fig_kwargs)

    return X_octave, bin_octave

#=============================================================================
def audio_SNR (s, mode ='fast', Nt=512) :
    """
    Computes the signal to noise ratio (SNR) of an audio signal in the time domain

    Parameters
    ----------
    s : 1D array
        Audio to process
    mode : str, optional, default is `fast`
        Select the mode to compute the envelope of the audio waveform
        `fast` : The sound is first divided into frames (2d) using the 
        function _wave2timeframes(s), then the max of each frame gives a 
        good approximation of the envelope.
        `Hilbert` : estimation of the envelope from the Hilbert transform. 
        The method is slow
    Nt : integer, optional, default is 512
        Size of each frame. The largest, the highest is the approximation.
    
    Returns
    -------
    ENRt : float
        Total energy in dB computed in the time domain
    BGNt : float
        Estimation of the background energy (dB) computed in the time domain
    SNRt: float
        Signal to noise ratio (dB) computed in the time domain 
        SNRt = ENRt - BGNt

    References
    ----------
    ..[1] Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms 
          Derived from Natural Recordings of the Environment. 
          Queensland University of Technology, Brisbane.
    ..[2] Towsey, Michael (2017),The calculation of acoustic indices derived 
          from long-duration recordings of the naturalenvironment.
          Queensland University of Technology, Brisbane.

    Examples
    --------
    >>> s, fs = maad.sound.load('guyana_tropical_forest.wav')
    >>> _,_,snr = maad.sound.audio_SNR(s)
    >>> snr
    1.5744987447774665

    """
    # Envelope
    env = envelope(s, mode=mode, Nt=Nt)
    # linear to power dB
    envdB = power2dB(env**2)
    # total energy estimation. 
    ENRt = mean_dB(envdB, axis=1)
    # Background noise estimation
    BGNt = get_unimode (envdB, mode ='median')
    # Signal to Noise ratio estimation
    SNRt = ENRt - BGNt

    return ENRt, BGNt, SNRt

#=============================================================================
def spectral_SNR (Sxx_power) :
    """
    Computes the signal to noise ratio (SNR) of an audio from its spectrogram
    in the time-frequency domain

    Parameters
    ----------
    Sxx_power : 2D array
        Power spectrogram density [Matrix] to process.
        
    Returns
    -------
    ENRf : float
        Total energy in dB computed in the frequency domain which corresponds
        to the average of the power spectrogram then the sum of the average
    BGNf : float
        Estimation of the background energy (dB) computed based on the estimation
        of the noise profile of the power spectrogram (2d)
    SNRf: float
        Signal to noise ratio (dB) computed in the frequency domain 
        SNRf = ENRf - BGNf
    ENRf_per_bin : vector of floats
        Energy in dB per frequency bin
    BGNf_per_bin : vector of floats
        Background (noise profile) energy in dB per frequency bin
    SNRf_per_bin : vector of floats  
        Signal to noise ratio per frequency bin
        
    References
    ----------
    ..[1] Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms 
          Derived from Natural Recordings of the Environment. 
          Queensland University of Technology, Brisbane.
    ..[2] Towsey, Michael (2017),The calculation of acoustic indices derived 
          from long-duration recordings of the naturalenvironment.
          Queensland University of Technology, Brisbane.

    Examples
    --------
    >>> s, fs = maad.sound.load('guyana_tropical_forest.wav')
    >>> Sxx_power,_,_,_ = maad.sound.spectrogram (s, fs)  
    >>> _, _, snr, _, _, _ = maad.sound.spectral_SNR(Sxx_power)
    >>> snr
    4.111567933859188
    
    """
    # average Sxx_power along time axis
    ENRf_per_bin = avg_power_spectro(Sxx_power)
    # compute total energy in dB
    ENRf = power2dB(sum(ENRf_per_bin))
    # Extract the noise profile (BGNf_per_bin) from the spectrogram Sxx_power
    _, noise_profile = remove_background_along_axis(Sxx_power, mode='median',axis=1) 
    # smooth the profile by removing spurious thin peaks (less than 5 pixels wide)
    noise_profile = running_mean(noise_profile,N=5)
    # noise_profile (energy) into dB
    BGNf_per_bin= power2dB(noise_profile)
    # compute noise/background energy in dB
    BGNf = power2dB(sum(noise_profile))
    # compute signal to noise ratio
    SNRf = ENRf - BGNf 
    # compute SNR_per_bin
    SNRf_per_bin = ENRf_per_bin - ENRf_per_bin

    return ENRf, BGNf, SNRf, ENRf_per_bin, BGNf_per_bin,SNRf_per_bin 


#=============================================================================
def select_bandwidth (s, fs, fcut, forder, fname ='butter', ftype='bandpass', 
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
    ftype : {`bandpass`, `lowpass`, `highpass`, `bandstop`}, optional, default 
        is `bandpass`
        The type of filter.
    fname : {`butter`, 'cheby1`, `cheby2`, `ellip`, `bessel`}, optional, default
        is 'butter'   
        The type of IIR filter to design:
        - Butterworth : `butter`
        - Chebyshev I : 'cheby1`
        - Chebyshev II : `cheby2`
        - Cauer/elliptic: `ellip`
        - Bessel/Thomson: `bessel`   
    rp : float, optional
        For Chebyshev and elliptic filters, provides the maximum ripple in 
        the passband. (dB)   
    rs : float, optional
        For Chebyshev and elliptic filters, provides the minimum attenuation 
        in the stop band. (dB)           
            
    Returns
    -------
    s_out : array_like
        The filtered output with the same shape and phase as s
        
    See Also
    --------
    fir_filter1d
        Lowpass, highpass, bandpass or bandstop a 1d signal with an Fir filter
    
    Examples
    --------
    Load and display the spectrogram of a sound waveform
    
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(w,fs)
    >>> Sxx_dB = maad.util.power2dB(Sxx_power) # convert into dB 
    >>> fig_kwargs = {'vmax': max(Sxx_dB),
                      'vmin':-90,
                      'extent':(tn[0], tn[-1], fn[0], fn[-1]),
                      'figsize':(4,13),
                      'title':'Power spectrogram density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    >>> fig1, ax1 = maad.util.plot2D(Sxx_dB, **fig_kwargs)
    
    Filter the waveform : keep the bandwidth between 6-10kHz
    
    >>> w_filtered = maad.sound.select_bandwidth(w,fs,fcut=(6000,10000), forder=5, fname ='butter', ftype='bandpass')
    >>> Sxx_power_filtered,tn,fn,_ = maad.sound.spectrogram(w_filtered,fs)
    >>> Sxx_dB_filtered = maad.util.power2dB(Sxx_power_filtered) # convert into dB 
    >>> maad.util.plot2D(Sxx_dB_filtered, **fig_kwargs)
    
    """
    sos = iirfilter(N=forder, Wn=np.asarray(fcut)/(fs/2), btype=ftype,ftype=fname, rp=rp, 
                     rs=rs, output='sos')
    # use sosfiltfilt insteasd of sosfilt to keep the phase of y matches s
    s_out = sosfiltfilt(sos, s)
    return s_out
 
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
    . power (mode='psd')
    . amplitude (mode = 'amplitude') => sqrt(power)
    . complex with real and imaginary parts (mode = 'complex')
    
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
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        - savefilename : str, optional, default :'_filt_audiogram.png'
            Postfix of the figure filename
        - db_range : scalar, optional, default : 100
            if db_range is a number, anything lower than -db_range is set to 
            -db_range and anything larger than 0 is set to 0
        - figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        - title : string, optional, default : 'Spectrogram'
            title of the figure
        - xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        - ylabel : string, optional, default : 'Amplitude [AU]'
        - cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
            'viridis'...
        - vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        
        - extent : list of scalars [left, right, bottom, top], optional, default: None
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
    Sxx : 2d ndarray of floats
        Spectrogram : Matrix containing K frames with N/2 frequency bins, 
        K*N <= length (wave)
        Sxx unit is power => Sxx_power if mode is 'psd'
        Sxx unit is amplitude => Sxx_ampli if mode is 'amplitude' or 'complex'
    tn : 1d ndarray of floats
        time vector (horizontal x-axis)    
    fn : 1d ndarray of floats
        Frequency vector (vertical y-axis)    
    extent : list of scalars [left, right, bottom, top]
        The location, in data-coordinates, of the lower-left and
        upper-right corners. 
    
    Notes
    -----
    This function take care of the energy conservation which is crucial 
    when working in with sound pressure level (dB SPL)
        
    Examples
    --------
    >>> s,fs = maad.sound.load("guyana_tropical_forest.wav")
    
    Compute energy of signal s
    
    >>> E1 = sum(s**2)
    >>> maad.util.power2dB(E1)
    44.861029507805256
    
    Compute the spectrogram with 'psd' output (if N<4096, the energy is lost)
    
    >>> N = 4096
    >>> Sxx_power,tn,fn,ext = maad.sound.spectrogram (s, fs, nperseg=N, noverlap=N//2, mode = 'psd')   
    
    Display Power Spectrogram
    
    >>> Sxx_dB = maad.util.power2dB(Sxx_power) # convert into dB
    >>> fig_kwargs = {'vmax': max(Sxx_dB),
                      'vmin':-70,
                      'extent':ext,
                      'figsize':(4,13),
                      'title':'Power spectrogram density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    fig, ax = maad.util.plot2D(Sxx_dB,**fig_kwargs)     
    
    Compute mean power spectrogram
    
    >>> S_power_mean = mean(Sxx_power, axis = 1)
    
    energy => power x time
    
    >>> E2 = sum(S_power_mean*len(s)) 
    >>> maad.util.power2dB(E2)
    44.93083283875093

    Compute the spectrogram with 'amplitude' output
    
    >>> Sxx_ampli,tn,fn,_ = maad.sound.spectrogram (s, fs, nperseg=N, noverlap=N//2, mode='amplitude')  
    
    For energy conservation => convert Sxx_ampli (amplitude) into power before doing the average.
    
    >>> S_power_mean = mean(Sxx_ampli**2, axis = 1)
    
    energy => power x time
    
    >>> E3 = sum(S_power_mean*len(s)) 
    >>>  maad.util.power2dB(E3)
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
    
    if verbose :
        print ('spectrogram dimension Nx=%d Ny=%d' % (Sxx_complex.shape[0], Sxx_complex.shape[1]))
    
    # Crop the image in order to analyzed only a portion of it
    if (fcrop or tcrop) is not None:
        if verbose:print ('Crop the spectrogram along time axis and frequency axis')
        Sxx_out, tn, fn = crop_image(Sxx_out,tn,fn,fcrop,tcrop)

    if verbose:
        print('max value of the spectrogram %.5f' % Sxx_out.max())

    # Extent
    extent = [tn[0], tn[-1], fn[0], fn[-1]]
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
             
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Spectrogram')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 0.33*(extent[1]-extent[0])))
        db_range=kwargs.pop('db_range',96)
        
        #### convert into dB 
        if mode=='psd':  
            Sxx_disp = power2dB(Sxx_out, db_range=db_range)
        if mode=='amplitude': 
            Sxx_disp = amplitude2dB(Sxx_out,db_range=db_range)
        if mode=='complex': 
            Sxx_disp = amplitude2dB(Sxx_out,db_range=db_range)
            
        vmin=kwargs.pop('vmin',-db_range) 
        vmax=kwargs.pop('vmax',Sxx_disp.max()) 

        _, fig = plot2D (Sxx_disp, extent=extent, figsize=figsize,title=title, 
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

    return Sxx_out, tn, fn, extent   

#=============================================================================
def avg_power_spectro (Sxx_power) :
    """
    Computes the average of a power spectrogram along the time axis
    
    Parameters
    ----------
    Sxx_power : 2d ndarray of floats
        Power spectrogram [Matrix]

    Returns
    -------
    S_power_mean : 1d ndarray of floats
        Power spectrum [Vector]
        
    See Also
    --------
    avg_amplitude_spectro
    
    """
    S_power_mean = np.mean(Sxx_power, axis=1)

    return S_power_mean

#=============================================================================
def avg_amplitude_spectro (Sxx_ampli) :
    """
    Computes the average of an amplitude spectrogram along the time axis
    
    Parameters
    ----------
    Sxx_ampli : 2d ndarray of floats
        Amplitude spectrogram [Matrix]
        
    Returns
    -------
    S_ampli_mean : 1d ndarray of floats
        Amplitude spectrum [Vector]
        
    See Also
    --------
    avg_power_spectro
    
    """
    
    # average the amplitude spectrogram taking the PSD for energy conservation
    S_ampli_mean = np.sqrt(np.mean(Sxx_ampli**2, axis=1))

    return S_ampli_mean