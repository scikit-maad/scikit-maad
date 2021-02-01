#!/usr/bin/env python
""" 
Collection of functions transform an audio signal into spectrogram and 
manipulate spectrograms.
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

# Import internal modules
from maad.util import (plot1d, plot2d, crop_image, power2dB, amplitude2dB)

# %%
# =============================================================================
# public functions
# =============================================================================


def spectrogram(x, fs, window='hann', nperseg=1024, noverlap=None,
                flims=None, tlims=None,
                mode='psd',
                verbose=False, display=False,
                savefig=None, **kwargs):
    """
    Compute a spectrogram using the short-time Fourier transform from an audio signal.

    The function can compute diferent outputs according to the parameter 'mode':
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
    flims, tlims : list of 2 scalars [min, max], optional, default is None
        flims corresponds to the min and max boundary frequency values
        tlims corresponds to the min and max boundary time values  
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
    This function takes care of the energy conservation which is crucial 
    when working in with sound pressure level (dB SPL)

    Examples
    --------
    >>> s,fs = maad.sound.load("../data/guyana_tropical_forest.wav")

    Compute energy of signal s

    >>> E1 = sum(s**2)
    >>> maad.util.power2dB(E1)
    44.861029507805256

    Compute the spectrogram with 'psd' output (if N<4096, the energy is lost)

    >>> N = 4096
    >>> Sxx_power,tn,fn,ext = maad.sound.spectrogram (s, fs, nperseg=N, noverlap=N//2, mode = 'psd')   

    Display Power Spectrogram

    >>> Sxx_dB = maad.util.power2dB(Sxx_power) # convert into dB
    >>> fig_kwargs = {'vmax': Sxx_dB.max(),
                      'vmin':-70,
                      'extent':ext,
                      'figsize':(4,13),
                      'title':'Power spectrogram density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    >>> fig, ax = maad.util.plot2d(Sxx_dB,**fig_kwargs)     

    Compute mean power spectrogram

    >>> S_power_mean = maad.sound.avg_power_spectro(Sxx_power)

    energy => power x time

    >>> E2 = sum(S_power_mean*len(s)) 
    >>> maad.util.power2dB(E2)
    44.93083283875093

    Compute the spectrogram with 'amplitude' output

    >>> Sxx_ampli,tn,fn,_ = maad.sound.spectrogram (s, fs, nperseg=N, noverlap=N//2, mode='amplitude')  

    For energy conservation => convert Sxx_ampli (amplitude) into power before doing the average.

    >>> S_ampli_mean = maad.sound.avg_amplitude_spectro(Sxx_ampli)
    >>> S_power_mean = S_ampli_mean**2

    energy => power x time

    >>> E3 = sum(S_power_mean*len(s)) 
    >>> maad.util.power2dB(E3)
    44.93083283875093

    """

    # Test if noverlap is None. By default, noverlap is half the length of the fft
    if noverlap is None:
        noverlap = nperseg // 2

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

    if mode == 'complex':
        Sxx_out = Sxx_complex

    if mode == 'amplitude':
        Sxx_out = Sxx

    if mode == 'psd':
        Sxx_out = PSDxx

    # test if the last frames are computed on a whole time frame.
    # if note => remove these frames
    if PSDxx.shape[1] > K:
        sup = Sxx_out.shape[1] - K
        Sxx_out = Sxx_out[:, :-sup]
        tn = tn[:-sup]

    # Remove the last frequency bin in order to obtain nperseg/2 frequency bins
    # instead of nperseg/2 + 1
    Sxx_out = Sxx_out[:-1, :]
    fn = fn[:-1]

    if verbose:
        print('spectrogram dimension Nx=%d Ny=%d' %
              (Sxx_complex.shape[0], Sxx_complex.shape[1]))

    # Crop the image in order to analyzed only a portion of it
    if (flims or tlims) is not None:
        if verbose:
            print('Crop the spectrogram along time axis and frequency axis')
        Sxx_out, tn, fn = crop_image(Sxx_out, tn, fn, flims, tlims)

    if verbose:
        print('max value of the spectrogram %.5f' % Sxx_out.max())

    # Extent
    extent = [tn[0], tn[-1], fn[0], fn[-1]]
    # dt and df resolution
    dt = tn[1]-tn[0]
    df = fn[1]-fn[0]

    if verbose == True:
        print("*************************************************************")
        print("   Time resolution dt=%.2fs | Frequency resolution df=%.2fHz "
              % (dt, df))
        print("*************************************************************")

    # Display
    if display:

        ylabel = kwargs.pop('ylabel', 'Frequency [Hz]')
        xlabel = kwargs.pop('xlabel', 'Time [sec]')
        title = kwargs.pop('title', 'Spectrogram')
        cmap = kwargs.pop('cmap', 'gray')
        figsize = kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
        db_range = kwargs.pop('db_range', 96)

        # convert into dB
        if mode == 'psd':
            Sxx_disp = power2dB(Sxx_out, db_range=db_range)
        if mode == 'amplitude':
            Sxx_disp = amplitude2dB(Sxx_out, db_range=db_range)
        if mode == 'complex':
            Sxx_disp = amplitude2dB(Sxx_out, db_range=db_range)

        vmin = kwargs.pop('vmin', -db_range)
        vmax = kwargs.pop('vmax', Sxx_disp.max())

        _, fig = plot2d(Sxx_disp, extent=extent, figsize=figsize, title=title,
                        ylabel=ylabel, xlabel=xlabel, vmin=vmin, vmax=vmax,
                        cmap=cmap, **kwargs)
        # SAVE FIGURE
        if savefig is not None:
            dpi = kwargs.pop('dpi', 96)
            bbox_inches = kwargs.pop('bbox_inches', 'tight')
            format = kwargs.pop('format', 'png')
            savefilename = kwargs.pop('savefilename', '_spectrogram')
            filename = savefig+savefilename+'.'+format
            print('\n''save figure : %s' % filename)
            fig.savefig(fname=filename, dpi=dpi, bbox_inches=bbox_inches,
                        format=format, **kwargs)

    return Sxx_out, tn, fn, extent

# %%


def linear_to_octave(X, fn, thirdOctave=True, display=False, **kwargs):
    """
    Transform a linear spectrum (1d) or Spectrogram (2d into octave or 1/3 octave
    spectrum (1d) or Spectrogram (2d).

    Our advice is to work with PSD (amplitude²) for energy conservation.

    Parameters
    ----------
    X : ndarray of floats
        Linear spectrum (1d) or Spectrogram (2d). 
        Work with PSD to be consistent with energy conservation
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

    Examples
    --------
    >>> w, fs = maad.sound.load('../data/guyana_tropical_forest.wav') 
    >>> Sxx_power,tn,fn, ext = maad.sound.spectrogram (w, fs, nperseg=8192)
    >>> maad.sound.linear_to_octave(Sxx_power, fn, display=True, extent=ext, vmin=-50)
    """

    # define the third octave or octave frequency vector in Hz.
    if thirdOctave:
        bin_octave = np.array([16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                               400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                               5000, 6300, 8000, 10000, 12500, 16000, 20000])  # third octave band.
    else:
        bin_octave = np.array(
            [16, 31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])  # octave

    # get the corresponding octave from fn
    bin_octave = bin_octave[(bin_octave >= np.min(fn))
                            & (bin_octave <= np.max(fn))]

    # Bins limit
    bin_octave_low = bin_octave/(2**0.1666666)
    bin_octave_up = bin_octave*(2**0.1666666)

    # select the indices corresponding to the frequency bins range
    X_octave = []
    for ii in np.arange(len(bin_octave)):
        ind = (fn >= bin_octave_low[ii]) & (fn <= bin_octave_up[ii])
        X_octave.append(np.sum(X[ind, ], axis=0))

    X_octave = np.asarray(X_octave)

    if display:
        X_octave_dB = power2dB(X_octave)
        if np.ndim(X_octave_dB) == 2:
            extent = kwargs.pop('extent', None)
            if extent is not None:
                xlabel = 'Time [sec]'
                duration = extent[1] - extent[0]
                deltaT = int(duration/10)
                xticks = (np.floor(np.arange(0, X.shape[1], X.shape[1]/10)*10)/10,
                          np.floor(np.arange(extent[0], extent[1], deltaT)*10)/10)
                figsize = (4, 0.33*(extent[1]-extent[0]))
            else:
                xlabel = 'pseudoTime [points]'
                xticks = np.arange(0, X.shape[1], 100),
                figsize = (4, 13)

            fig_kwargs = {'vmax': kwargs.pop('vmax', np.max(X_octave_dB)),
                          'vmin': kwargs.pop('vmin', np.min(X_octave_dB)),
                          'xticks': xticks,
                          'figsize': kwargs.pop('figsize', figsize),
                          'yticks': (np.arange(len(bin_octave)), bin_octave),
                          'title': 'Octave Spectrogram',
                          'xlabel': xlabel,
                          'ylabel': 'Frequency [Hz]',
                          }
            plot2d(X_octave_dB, **fig_kwargs)
        elif np.ndim(X_octave_dB) == 1:
            fig_kwargs = {
                'title': 'Octave Spectrum',
                'xlabel': kwargs.pop('xlabel', 'Frequency [Hz]'),
                'ylabel': kwargs.pop('ylabel', 'Amplitude [dB]'),
            }
            plot1d(bin_octave, X_octave_dB, **fig_kwargs)

    return X_octave, bin_octave

# %%


def avg_power_spectro(Sxx_power):
    """
    Computes the average of a power spectrogram along the time axis.

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

    Examples
    --------
    >>> w, fs = maad.sound.load('../data/guyana_tropical_forest.wav') 
    >>> Sxx_power,tn, fn, ext = maad.sound.spectrogram (w, fs)

    Compute mean power spectrogram

    >>> S_power_mean = maad.sound.avg_power_spectro(Sxx_power)
    >>> S_power_mean
    array([2.91313297e-04, 9.16802665e-04, 2.89179556e-04, 1.04767281e-04,
       6.66154492e-05, 4.59636926e-05, 3.59688131e-05, 3.30869371e-05,
       ...


    """
    S_power_mean = np.mean(Sxx_power, axis=1)

    return S_power_mean

# %%

def avg_amplitude_spectro(Sxx_ampli):
    """
    Computes the average of an amplitude spectrogram along the time axis.

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

    Examples
    --------
    >>> w, fs = maad.sound.load('../data/guyana_tropical_forest.wav') 
    >>> Sxx_amplitude,tn, fn, ext = maad.sound.spectrogram (w, fs, mode="amplitude")

    Compute mean amplitude spectrogram

    >>> S_amplitude_mean = maad.sound.avg_amplitude_spectro(Sxx_amplitude)
    >>> S_amplitude_mean
    array([0.0170679 , 0.03027875, 0.01700528, 0.01023559, 0.00816183,
       0.00677965, 0.0059974 , 0.00575212, 0.00700752, 0.00926279,
       ...

    """

    # average the amplitude spectrogram taking the PSD for energy conservation
    S_ampli_mean = np.sqrt(np.mean(Sxx_ampli**2, axis=1))

    return S_ampli_mean
