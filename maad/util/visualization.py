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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys

# min value
import sys
_MIN_ = sys.float_info.min


#=============================================================================
def plot1D(x, y, ax=None, **kwargs):
    """
    plot a signal s
    
    Parameters
    ----------
    x : 1d ndarray of integer
        Vector containing the abscissa values (horizontal axis)
             
    y : 1d ndarray of scalar
        Vector containing the ordinate values (vertical axis)  
    
    ax : axis, optional, default is None
        Draw the signal on this specific axis. Allow multiple plots on the same
        axis.
            
    \*\*kwargs, optional
        
        - figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        
        - facecolor : matplotlib color, optional, default: 'w' (white)
            the background color.  
        
        - edgecolor : matplotlib color, optional, default: 'k' (black)
            the border color. 
        
        - linecolor : matplotlib color, optional, default: 'k' (black)
            the line color
        
        The following color abbreviations are supported:
    
        ==========  ========
        character   color
        ==========  ========
        'b'         blue
        'g'         green
        'r'         red
        'c'         cyan
        'm'         magenta
        'y'         yellow
        'k'         black
        'w'         white
        ==========  ========
        
        In addition, you can specify colors in many ways, including RGB tuples 
        (0.2,1,0.5). See matplotlib color 
        
        - linewidth : scalar, optional, default: 0.5
            width in pixels
        
        - figtitle: string, optional, default: 'Audiogram'
            Title of the plot 
        
        - xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        
        - ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        
        - legend : string, optional, default : None
            Legend for the plot
        
        - now : boolean, optional, default : True
            if True, display now. Cannot display multiple plots. 
            To display mutliple plots, set now=False until the last call for 
            the last plot         
        
        ...and more, see matplotlib
        
    Returns
    -------
    fig : Figure
        The Figure instance 
    ax : Axis
        The Axis instance
        
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> p = maad.util.wav2pressure (w, gain=42)
    >>> Pxx,tn,fn,_ = maad.sound.spectrogram(p,fs)
    >>> Lxx = maad.util.power2dBSPL(Pxx) # convert into dB SPL
       
    Plot the spectrum at t = 7s
    
    >>> index = maad.util.nearest_idx(tn,7)
    >>> fig_kwargs = {'figtitle':'Spectrum (PSD)',
                      'xlabel':'Frequency [Hz]',
                      'ylabel':'Power [dB]',
                      }
    
    >>> fig, ax = maad.util.plot1D(fn, Lxx[:,index], **fig_kwargs)
    """  

    figsize=kwargs.pop('figsize', (4, 10))
    facecolor=kwargs.pop('facecolor', 'w')
    edgecolor=kwargs.pop('edgecolor', 'k')
    linewidth=kwargs.pop('linewidth', 0.1)
    linecolor=kwargs.pop('linecolor', 'k')
    title=kwargs.pop('figtitle', 'Audiogram')
    xlabel=kwargs.pop('xlabel', 'Time [s]')
    ylabel=kwargs.pop('ylabel', 'Amplitude [AU]')
    legend=kwargs.pop('legend', None)
    now=kwargs.pop('now', True)
       
    # if no ax, create a figure and a subplot associated a figure otherwise
    # find the figure that belongs to ax
    if ax is None :
        fig, ax = plt.subplots()
        # set the paramters of the figure
        fig.set_figheight(figsize[0])
        fig.set_figwidth (figsize[1])
        fig.set_facecolor(facecolor)
        fig.set_edgecolor(edgecolor)
    else:
        fig = ax.get_figure()
    
    # plot the data on the subplot
    line = ax.plot(x, y, linewidth, color=linecolor)
    
    # set legend to the line
    line[0].set_label(legend)
    
    # set the parameters of the subplot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis('tight')
    ax.grid(True)
    if legend is not None: ax.legend()
    
    # Display the figure now
    if now : plt.show()

    return ax, fig

#=============================================================================

def plot2D(im,ax=None,**kwargs):
    """
    display an image (spectrogram, 2D binary mask, ROIS...)
    
    Parameters
    ----------
    im : 2D numpy array 
        The name or path of the .wav file to load
             
    ax : axis, optional, default is None
        Draw the image on this specific axis. Allow multiple plots on the same
        axis.
            
    \*\*kwargs, optional
        
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
        
        - now : boolean, optional, default : True
            if True, display now. Cannot display multiple images. 
            To display mutliple images, set now=False until the last call for 
            the last image      
            
        ... and more, see matplotlib
        
    Returns
    -------
    fig : Figure
        The Figure instance 
    ax : Axis
        The Axis instance
        
    Examples
    --------
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
        
    """ 
   
    # matplotlib parameters
    figsize=kwargs.pop('figsize', (4, 13))
    title=kwargs.pop('title', 'Spectrogram')
    ylabel=kwargs.pop('ylabel', 'Frequency [Hz]')
    xlabel=kwargs.pop('xlabel', 'Time [sec]')  
    cmap=kwargs.pop('cmap', 'gray') 
    vmin=kwargs.pop('vmin', None) 
    vmax=kwargs.pop('vmax', None)    
    ext=kwargs.pop('extent', None)   
    now=kwargs.pop('now', True)
    
    # if no ax, create a figure and a subplot associated a figure otherwise
    # find the figure that belongs to ax
    if ax is None :
        fig, ax = plt.subplots()
        # set the paramters of the figure
        fig.set_facecolor('w')
        fig.set_edgecolor('k')
        fig.set_figheight(figsize[0])
        fig.set_figwidth (figsize[1])
    else:
        fig = ax.get_figure()

    # display image
    _im = ax.imshow(im, extent=ext, interpolation='none', origin='lower', vmin =vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(_im, ax=ax)

    # set the parameters of the subplot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis('tight') 

    fig.tight_layout()
    
    # Display the figure now
    if now: plt.show()

    return ax, fig

#=============================================================================
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    
    Parameters
    ----------
    nlabels : Integer
        Number of labels (size of colormap)
    type : String
        'bright' for strong colors, 'soft' for pastel colors. Default is 'bright'
    first_color_black : Boolean
        Option to use first color as black. Default is True  
    last_color_black : Boolean   
        Option to use last color as black, Default is False
    verbose  : Boolean   
        Prints the number of labels and shows the colormap. Default is False
    
    Returns
    -------
    random_colormap : Colormap 
        Colormap type used by matplotlib

    References
    ----------
    adapted from https://github.com/delestro/rand_cmap author : delestro    
    """
    
    # initialize the random seed in order to get always the same random order
    np.random.seed(seed=321)
    
    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.1, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.5, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

    if first_color_black:
        randRGBcolors[0] = [0, 0, 0]

    if last_color_black:
        randRGBcolors[-1] = [0, 0, 0]
            
    random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

   
    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

#=============================================================================
def crop_image (im, tn, fn, fcrop=None, tcrop=None):
    """
    Crop a spectrogram (or an image) in time (horizontal X axis) and frequency
    (vertical y axis)
    
    Parameters
    ----------
    im : 2d ndarray
        image to be cropped
    
    tn, fn : 1d ndarray
        tn is the time vector. fn is the frequency vector. They are required 
        in order to know the correspondance between pixels and (time,frequency)
    
    fcrop, tcrop : list of 2 scalars [min, max], optional, default is None
        fcrop corresponds to the min and max boundary frequency values
        tcrop corresponds to the min and max boundary time values
                
    Returns
    -------
    im : 2d ndarray
        image cropped
        
    tn, fn, 1d ndarray
        new time and frequency vectors
        
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> p = maad.util.wav2pressure (w, gain=42)
    >>> Pxx,tn,fn,_ = maad.sound.spectrogram(p,fs)
    >>> Lxx = maad.util.power2dBSPL(Pxx) # convert into dB SPL
    >>> fig_kwargs = {'vmax': max(Lxx),
                      'vmin':0,
                      'figsize':(10,13),
                      'extent':(tn[0], tn[-1], fn[0], fn[-1]),
                      'title':'Power spectrogram density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    >>> fig, ax = maad.util.plot2D(Lxx,**fig_kwargs)      
    
    >>> Lxx_crop, tn_crop, fn_crop = maad.util.crop_image(Lxx, tn, fn, fcrop=(2000,6000), tcrop=(0,30))
    >>> fig_kwargs = {'vmax': max(Lxx),
                      'vmin':0,
                      'figsize':(10*len(fn_crop)/len(fn),13*len(tn_crop)/len(tn)),
                      'extent':(tn_crop[0], tn_crop[-1], fn_crop[0], fn_crop[-1]),
                      'title':'Crop of the power spectrogram density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    >>> fig, ax = maad.util.plot2D(Lxx_crop,**fig_kwargs)  
    """
    
    if tcrop is not None : 
        indt = (tn>=tcrop[0]) *(tn<=tcrop[1])
        im = im[:, indt]
        # redefine tn
        tn = tn[np.where(indt>0)]
    if fcrop is not None : 
        indf = (fn>=fcrop[0]) *(fn<=fcrop[1])
        im = im[indf, ]
        # redefine fn
        fn = fn[np.where(indf>0)]
    
    return im, tn, fn
