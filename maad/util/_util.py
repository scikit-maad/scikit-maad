#!/usr/bin/env python
""" Utilitary functions for scikit-MAAD """
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: BSD 3 clause

# =============================================================================
# Load the modules
# =============================================================================
# Import external modules
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys
from datetime import datetime
import pandas as pd


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

def read_audacity_annot (audacity_filename):
    """
    Read audacity annotations file (or labeling file) and return a Pandas Dataframe
    with the bounding box and the label of each region of interest (ROI)
    
    Parameters
    ----------
    audacity_filename : String
        Path to the audacity file

    Returns
    -------
    tab_out : Pandas Dataframe 
        Colormap type used by matplotlib
    
    References
    ----------
    https://manual.audacityteam.org/man/label_tracks.html   
    """
    # read file with tab delimiter
    tab_in = pd.read_csv(audacity_filename, delimiter='\t', header=None)
    
    # arrange data
    t_info = tab_in.loc[np.arange(0,len(tab_in),2),:]
    t_info = t_info.rename(index=str, columns={0: 'tmin', 1: 'tmax', 2:'label'})
    t_info = t_info.reset_index(drop=True)
    
    f_info = tab_in.loc[np.arange(1,len(tab_in)+1,2),:]
    f_info = f_info.rename(index=str, columns={0: 'slash', 1: 'fmin', 2:'fmax'})
    f_info = f_info.reset_index(drop=True)
    
    # return dataframe
    tab_out = pd.concat([t_info['label'], 
                         t_info['tmin'].astype('float32'), 
                         f_info['fmin'].astype('float32'), 
                         t_info['tmax'].astype('float32'), 
                         f_info['fmax'].astype('float32')],  axis=1)

    return tab_out



def date_from_filename (filename):
    """
    Extract date and time from the filename. Return a datetime object
    
    Parameters
    ----------
    filename : string
        The filename must follow this format :
            XXXX_yyyymmdd_hhmmss.wav
            with yyyy : year / mm : month / dd: day / hh : hour (24hours) /
            mm : minutes / ss : seconds
            
    Returns
    -------
    date : object datetime
    """
    # date by default
    date = datetime(1900,1,1,0,0,0,0)
    # test if it is possible to extract the recording date from the filename
    if filename[-19:-15].isdigit(): 
        yy=int(filename[-19:-15])
    else:
        return date
    if filename[-15:-13].isdigit(): 
        mm=int(filename[-15:-13])
    else:
        return date
    if filename[-13:-11].isdigit(): 
        dd=int(filename[-13:-11])
    else:
        return date
    if filename[-10:-8].isdigit(): 
        HH=int(filename[-10:-8])
    else:
        return date
    if filename[-8:-6].isdigit(): 
        MM=int(filename[-8:-6])
    else:
        return date
    if filename[-6:-4].isdigit(): 
        SS=int(filename[-6:-4])
    else:
        return date

    # extract date and time from the filename
    date = datetime(year=yy, month=mm, day=dd, hour=HH, minute=MM, second=SS, 
                    microsecond=0)
    
    return date

def linear_scale(datain, minval= 0.0, maxval=1.0):
    """ 
    Program to scale the values of a matrix from a user specified minimum to a user specified maximum
    
    Parameters
    ----------
    datain : array-like
        numpy.array like with numbers
    minval : scalar, optional, default : 0
        This minimum value is attributed to the minimum value of the array 
    maxval : scalar, optional, default : 1
        This maximum value is attributed to the maximum value of the array         
        
    Returns
    -------
    dataout : array-like
        numpy.array like with numbers  
        
    -------
    Example
    -------
        a = np.array([1,2,3,4,5]);
        a_out = scaledata(a,0,1);
    Out: 
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        
    References:
    ----------
    Program written by Aniruddha Kembhavi, July 11, 2007 for MATLAB
    Adapted by S. Haupert Dec 12, 2017 for Python
    """

    dataout = datain - datain.min();
    dataout = (dataout/dataout.max())*(maxval-minval);
    dataout = dataout + minval;
    return dataout

def db_scale (datain, db_range=60, db_gain=None):
    """
    Rescale the data in dB scale after normalizing the data
    
    Parameters
    ----------
    datain : array-like
        data to rescale in dB 
    db_range : scalar, optional, default : 60
        Anything larger than db_range is set to db_range  
    db_gain : scalar, optional, default is None
        Gain added to the results 
                --> 20*log10(a) + db_gain
                
    Returns
    -------
    dataout : scalars
        --> 20*log10(a) + db_gain 
    """
    
    
    datain = abs(datain)            # take the absolute value of datain
    datain[datain ==0] = 1e-32      # Avoid zero value for log10          
    dataout = 20*np.log10(datain)   # take log
    if db_gain : dataout = dataout + db_gain    # Add gain if needed
    
    # set anything less than the db_range as 
    #dataout[dataout < 0] = 0 
    dataout[dataout > 0] = 0  
    dataout[dataout < -(db_range)] = -db_range  
        
    return dataout

def crop_image (im, tn, fn, fcrop=None, tcrop=None):
    
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
            
    **kwargs, optional
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        facecolor : matplotlib color, optional, default: 'w' (white)
            the background color.  
        edgecolor : matplotlib color, optional, default: 'k' (black)
            the border color. 
        linecolor : matplotlib color, optional, default: 'k' (black)
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
        
        linewidth : scalar, optional, default: 0.5
            width in pixels
        figtitle: string, optional, default: 'Audiogram'
            Title of the plot 
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        legend : string, optional, default : None
            Legend for the plot
        
        ... and more, see matplotlib
    Returns
    -------
        fig : Figure
            The Figure instance 
        ax : Axis
            The Axis instance
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
    plt.show()

    return ax, fig

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
            
    **kwargs, optional
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
        ext : scalars (left, right, bottom, top), optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        ... and more, see matplotlib
    Returns
    -------
        fig : Figure
            The Figure instance 
        ax : Axis
            The Axis instance
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
    plt.show()

    return ax, fig

