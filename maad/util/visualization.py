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
import pandas as pd
from skimage.io import imsave 
import colorsys
from ast import literal_eval

# min value
import sys
_MIN_ = sys.float_info.min


#### Importation from internal modules
from maad.util import linear_scale

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
    color=kwargs.pop('color', 'k')
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
    line = ax.plot(x, y, linewidth, color=color, **kwargs)
    
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
        Image or Spectrogram
             
    ax : axis, optional, default is None
        Draw the image on this specific axis. Allow multiple plots on the same
        figure.
            
    \*\*kwargs, optional
        
        - figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        
        - title : string, optional, default : 'Spectrogram'
            title of the figure
        
        - xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        
        - ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
            
        - xticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should 
            be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place 
            at the given locs.
            
        - yticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should 
            be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place 
            at the given locs.
        
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
    title=kwargs.pop('title', 'Spectrogram')
    ylabel=kwargs.pop('ylabel', 'Frequency [Hz]')
    xlabel=kwargs.pop('xlabel', 'Time [sec]')  
    xticks = kwargs.pop('xticks', None) 
    yticks = kwargs.pop('yticks', None)
    cmap=kwargs.pop('cmap', 'gray') 
    vmin=kwargs.pop('vmin', None) 
    vmax=kwargs.pop('vmax', None)    
    extent=kwargs.pop('extent', None)   
    now=kwargs.pop('now', True)
    
    if extent is not None :
        figsize=kwargs.pop('figsize',(4, 0.33*(extent[1]-extent[0])))  
    else:
        figsize=kwargs.pop('figsize',(4, 13))  
    
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
    _im = ax.imshow(im, extent=extent, interpolation='none', origin='lower', 
                    vmin =vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(_im, ax=ax)

    # set the parameters of the subplot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None :  
        ax.set_xticks(ticks =xticks[0])
        ax.set_xticklabels(labels=xticks[1])
    if yticks is not None :  
        ax.set_yticks(ticks =yticks[0])
        ax.set_yticklabels(labels=yticks[1])
    ax.axis('tight') 

    fig.tight_layout()
    
    # Display the figure now
    if now: plt.show()

    return ax, fig

#=============================================================================
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, 
              verbose=False):
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

#=============================================================================
def save_figlist(fname, figlist):
    """
    Save a list of figures to file.
    
    Parameters
    ----------
    fname: string
        suffix name to save the figure. Extension indicates the format 
        of the image

    Returns
    -------
    Nothing
        
    """
    for i, fig in enumerate(figlist):
        fname_save='%d_%s' % (i, fname)
        imsave(fname_save,fig)

#=============================================================================
def plot_features_map (df, norm=True, mode='24h', **kwargs) :
    """
    Plot features amplitude as heatmap.
    Features are on the vertical axis
    Timeline is on the horizontal axis
    
    Parameters
    ----------
    df : Panda DataFrame
        DataFrame with features (ie. indices).
    
    norm : boolean, default is True
        if True, the features are scaled between 0 to 1
        
    mode : string in {'24h'}, default is '24h'
        Select if the timeline of the phenology :
            -'24h' : average of the results over a day
            - otherwise, the timeline is the timeline of the dataframe
            
    **kwargs
        SPECIFIC TO THIS FUNCTION
        - ftime : Time format to display as x label 
            by default '%Y-%m-%d'
            see https://docs.python.org/fr/3.6/library/datetime.html?highlight=strftime#strftime-strptime-behavior
            
        SPECIFIC TO MATPLOTLIB    
        - figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        
        - title : string, optional, default : 'Spectrogram'
            title of the figure
        
        - xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        
        - ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
            
        - xticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should 
            be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place 
            at the given locs.
            
        - yticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should 
            be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place 
            at the given locs.
        
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
    """

    if isinstance(df, pd.DataFrame) == False :
        raise TypeError ('df must be a Pandas Dataframe')
    elif isinstance(df.index, pd.DatetimeIndex) == False :
        raise TypeError ('df must have an index of type DateTimeIndex')
       
    if mode == '24h' :      
        # Mean values by hour
        df = df.groupby(df.index.hour).mean()
        # Get the list of unique index of type 'hour'
        x_label = [i + j for i, j in zip(map(str, df.index.values), ['h']*len(df))]
    else :
        # Get the list of unique index of type 'hour'
        ftime = kwargs.pop('ftime','%Y-%m-%d')
        x_value = df.index.strftime(ftime)
        x_label = x_value.tolist()
        
    if norm :
        df = linear_scale(df)
    
    # kwargs
    cmap = kwargs.pop('cmap','RdBu_r')  
    
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    caxes = ax.matshow(df.transpose(), cmap=cmap, aspect='auto',**kwargs)
    fig.colorbar(caxes, shrink=0.75, label='Normalized value')
    # Set ticks on both sides of axes on
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    # We want to show all ticks...
    ax.set_yticks(np.arange(len(df.columns)))
    ax.set_xticks(np.arange(len(x_label)))
    # ... and label them with the respective list entries
    ax.set_yticklabels(df.columns)
    ax.set_xticklabels(x_label)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
    fig.tight_layout()
    plt.show()
    
    return fig, ax

#=============================================================================

def plot_features (df, ax=None, norm=True, mode='24h', **kwargs) : 
    """
    Plot phenology of the features (ie. indices) in the DataFrame obtained 
    with MAAD
            
    Parameters
    ----------
    df : Panda DataFrame
        DataFrame with features (ie. indices).
    
    norm : boolean, default is True
        if True, the features are normalized by the max
        
    mode : string in {'24h'}, default is '24h'
        Select if the timeline of the phenology :
            -'24h' : average of the results over a day
            - otherwise, the timeline is the timeline of the dataframe
            
    **kwargs
        - figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        
        - figtitle : string, optional, default : ''
            title of the figure
        
        - xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        
        - ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
            
        - xticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should 
            be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place 
            at the given locs.
            
        - yticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should 
            be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place 
            at the given locs.
        
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
    """
    if isinstance(df, pd.DataFrame) == False :
        raise TypeError ('df must be a Pandas Dataframe')
    elif isinstance(df.index, pd.DatetimeIndex) == False :
        raise TypeError ('df must have an index of type DateTimeIndex')
        
    if mode == '24h' :      
        # Mean values by hour
        df = df.groupby(df.index.hour).mean()
        
    if norm :
        df = linear_scale(df)

    # plot
    import itertools
    from matplotlib.lines import Line2D
    list_markers = tuple(list(Line2D.markers.keys())[0:-4])
    markers = itertools.cycle(list_markers) 
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = itertools.cycle(prop_cycle.by_key()['color'])
    
    figsize = kwargs.pop('figsize',(5,5))
    kwargs.pop('label',None)
    
    # if no ax, create a figure and a subplot associated a figure otherwise
    # find the figure that belongs to ax
    if ax is None :
        fig, ax = plt.subplots(**kwargs)
        fig.set_size_inches(figsize)
    else:
        fig = ax.get_figure()

    for indice in list(df) :
        ax.plot(df.index, df[indice], marker=next(markers), color=next(colors),
                linestyle='-', label=indice, **kwargs) 
        
    if mode == '24h' :  
        ax.set_xlabel('Day time (Hour)')
    ax.grid()
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    return fig, ax

#=============================================================================
def plot_correlation_map (df, R_threshold=0.75, method ='spearman', **kwargs) :
    """
    Plot the correlation map between indices in the DataFrame obtained
    with MAAD
        
    Parameters
    ----------
    df : Panda DataFrame
        DataFrame with features (ie. indices).
        
    R_threshold : scalar between 0 to 1, default is 0.75
        Show correlations with R higher than R_threshold
        
    method : string in {'spearman', 'pearson'}, default is 'spearman'
        Choose the correlation type
    
    **kwargs
        - figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        
        - title : string, optional, default : 'Spectrogram'
            title of the figure
        
        - xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        
        - ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
            
        - xticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should 
            be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place 
            at the given locs.
            
        - yticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should 
            be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place 
            at the given locs.
        
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
    """
    # Correlation matrix
    corr_matrix = df.corr(method)
     
    # pop kwargs
    figsize = kwargs.pop('figsize',(10,8))
    cmap = kwargs.pop('cmap','RdBu_r')
    label_colorbar = kwargs.pop('label_colorbar','R')
    
    fig = plt.figure()
    fig.set_size_inches(figsize)
    ax = fig.add_subplot(111)
    caxes = ax.matshow(corr_matrix[abs(corr_matrix)**2>R_threshold**2], cmap=cmap)
    fig.colorbar(caxes, shrink=0.75, label=label_colorbar)
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(corr_matrix.columns)
    ax.set_yticklabels(corr_matrix.columns)
    
    # Rotate the tick labels, set their alignment and fontsize
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=7)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=8)
    
    fig.tight_layout()
    plt.show()
    
    return fig, ax

#=============================================================================
def false_Color_Spectro (df, indices=None, plim=(1,99), reverseLUT=False, 
                         permut=False, unit='minutes', verbose=False,
                         display=False, savefig=None, **kwargs) :
    """
    Create False Color Spectrogram from indices obtained by MAAD
    Only indices than can be computed bin per bin (ie frequency per frequency)
    are used to create False Color Spectro. They are called xxx_per_bin
        
    Parameters
    ----------
    df : Panda DataFrame
        DataFrame with indices per frequency bin.
        
    indices : list, default is None
        List of indices. 
        If permut is False (see permut), 
            if indices is None : 1st indices is red (R), 2nd indice is green (G) 
                                 and the 3rd indices is blue (B).
            if indices is a list of 3 indices or more, only the 3 first indices
            (triplet) are used to create the false color spectro.
        if permut is True,
            if indices is None : All indices in df are used
            if indices is a list of 3 indices or more, all the indices in the 
            list are used to create the false color spectro with all the 
            possibile permutations.
    
    plim : list of 2 scalars, default is (1,99)
        Set the minimum and maximum percentile to define the min and max value 
        of each indice. These values are then used to clip the values of each 
        indices between these limits.
             
    reverseLUT: boolean, default is False
        LUT : look-up table.
        if False, the highest indice value is the brigthest color (255) 
        and the lowest indice value is the darkest (0)
        if True, it's the reverse order, the highest indice value is the 
        darkest color (0) and the lowest is the brighest (255)     
    
    permut : boolean, default is False
        if True, all the permutations possible for red, green, blue are computed
        among the list of indices (see indices)
    
    unit : string in {'minutes', 'hours', 'days', 'weeks'}, default is 'minutes'
        Select the unit base for x axis
        
    verbose : boolean, default is False
        print indices on the default terminal
        
    display : boolean, default is False
        Display False Color Spectro
        
    savefig : string, optional, default is None 
        if not None, figures will be safe. Savefig is the prefix of the save
        filename.
    
    \*\*kwargs, optional
       - dpi : scalar, optional, default 96
       - format : str, optional, default .png
    
    Returns
    -------
    false_color_image : ndarray of scalars
        Matrix with ndim = 4 if multiple false color spectro or ndim = 3, if
        single false color spectro with 3 colors : R, G, B
            
    triplet : list
        List of triplet of indices corresponding to each false color spectro
    """
    # sort dataframe by date
    df = df.sort_index(axis=0)
    # set index as DatetimeIndex
    df = df.set_index(pd.DatetimeIndex(df.index))
    # remove column file
    if 'file' in df.columns :
        df = df.drop(columns='file')
    # test if frequencies is in columns
    if 'frequencies' in df.columns :
        # get frequencies and remove the column
        # test if type of df['frequencies'] is number or str
        if isinstance(df.frequencies.iloc[0], str) :
            fn = df['frequencies'].apply(literal_eval).iloc[0]
        else :
            fn = df['frequencies'].iloc[0]
        # drop frequencies
        df = df.drop(columns='frequencies')

    # Set the list of indices if indices is None
    if indices is None :
        indices = list(df)
    
    # for each indice, check if values type is string and convert into scalars
    # Values are vectors of strings when data are loading from csv
    for indice in indices:
        # test if type of df[indice] is string
        if isinstance(df[indice].iloc[0], str) :
            # convert string into scalars
            df[indice] = df[indice].apply(literal_eval)
    
    # create a dataframe with normalized values for each indices
    df_z = pd.DataFrame()

    for indice in indices: 
        z = []
        if verbose : print(indice)
        for v in df[indice]:
            z.append(v)
        z = np.asarray(z).T
        #Select the min and max value for each indice
        z_min= np.percentile(z,plim[0])
        z_max = np.percentile(z,plim[1])
        # clip the value to the min and max found
        z = np.clip(z, z_min, z_max)
        # linear conversion 
        if reverseLUT == True :
            # between 1 to 0
            z = linear_scale(z,1,0)
        else:
            # between 0 to 1
            z = linear_scale(z,0,1)
            
        df_z[indice] = [z*255]
        
    # find all permutation of 3 indices among all indices
    if permut == True :
        import itertools
        per = itertools.permutations(list(df_z), 3)
        triplet = []
        for val in per:
            triplet.append([*val])
    else :
        triplet = []
        triplet.append(indices[0])
        triplet.append(indices[1])
        triplet.append(indices[2])
        triplet = [triplet]
        
    #####################
    
    # get the number of pixels along frequency (Nf) and time (Nt)
    Nf, Nt = df_z[triplet[0][0]].values.tolist()[0].shape
    
    # test if figsize is in kwargs
    figsize=kwargs.pop('figsize', None) 
    # if figsize is not in kwargs
    if figsize is None :
        figsize =  (6*Nt/250, 10*Nf/512)
        
    fig_kwargs = {'figsize': figsize,
                  'tight_layout':'tight_layout'}
    
    # number of days in the period
    deltaT = (df.index.max()-df.index.min())
    
    # unit
    if unit == 'minutes' :
        normT = 60e9
        xlabel = 'Minutes'
    elif unit == 'hours' :
        normT = 60e9*60
        xlabel = 'Hours'
    elif unit == 'days' :
        normT = 60e9*60*24
        xlabel = 'Days'
    elif unit =='weeks' :
        normT = 60e9*60*24*7
        xlabel = 'Weeks'
    else:
        normT = 60e9
        xlabel = 'Minutes'
    
    false_color_image =[]
    for tt in np.arange(len(triplet)) : 
        
        # create the false color image (R,G,B)
        z0 = df_z[triplet[tt][0]].values.tolist()[0]
        z1 = df_z[triplet[tt][1]].values.tolist()[0]
        z2 = df_z[triplet[tt][2]].values.tolist()[0]
        false_color_image.append((np.dstack((z0,z1,z2) )).astype(np.uint8))
        
        # Display the False Color Spectro
        if display :
            plt.rcParams.update({'font.size': 10})
            plt.rcParams.update({'font.family' : 'serif'})
            fig=plt.figure(facecolor='white', **fig_kwargs)
            plt.imshow(false_color_image[tt], aspect='auto', origin='lower', interpolation='none', extent=(fn[0], deltaT.value/normT, 0, fn[-1]))
            plt.xlabel(xlabel)
            plt.ylabel('Frequency (Hz)')
            plt.title('False Color Spectro ' + '\n'
                      + ' [R:'+ triplet[tt][0][:-8] + '; ' 
                      + 'G:'  + triplet[tt][1][:-8] + '; '
                      + 'B:'  + triplet[tt][2][:-8] + ']', size=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
    
            # SAVE FIGURE 
            if savefig is not None :              
                dpi=kwargs.pop('dpi', 96)  
                bbox_inches='tight'
                format=kwargs.pop('format','png')  
                filename = '_fcs_' + triplet[tt][0][:-8] +'_' + triplet[tt][1][:-8] +'_'+ triplet[tt][2][:-8]
                full_filename = savefig + filename + '.' + format 
                if verbose : print('\n''save figure : %s' %full_filename) 
                # save fig
                fig.savefig(fname=full_filename, dpi=dpi, bbox_inches=bbox_inches, 
                            format=format, **kwargs)    
                # close fig
                plt.close(fig)
                
    # convert into ndarray
    false_color_image= np.asarray(false_color_image)
    
    # test if there is only one False Color Spectro, then, remove the 1st dim
    if false_color_image.shape[0] == 1:
        false_color_image = false_color_image[0]
        triplet = triplet[0]
    
    return false_color_image, triplet


