#!/usr/bin/env python
""" 
Ensemble of functions that facilitate data visualization.
"""

# Import external modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
import pandas as pd
from skimage.io import imsave
import colorsys
from ast import literal_eval
from datetime import datetime

# min value
import sys

_MIN_ = sys.float_info.min

# Importation from internal modules
from maad.util import linear_scale, power2dB

#%%
def _check_axes(axes):
    """Check if "axes" is an instance of an axis object. If not, use `gca`."""
    if axes is None:
        import matplotlib.pyplot as plt

        axes = plt.gca()
    elif not isinstance(axes, Axes):
        raise ValueError(
            "`axes` must be an instance of matplotlib.axes.Axes. "
            "Found type(axes)={}".format(type(axes))
        )
    return axes


#%%
def _is_single_value(value):
    """
    Check if a value is a single numeric value or a single character.

    Parameters:
    -----------
        value (int, float, str): The value to be checked.

    Returns:
    --------
        bool: True if the value is a single numeric value or a single character, False otherwise.
    """
    if isinstance(value, (int, float, str)):
        if isinstance(value, (int, float)):
            return True  # It's a numeric value
        elif isinstance(value, str):
            # Check if it's a single numeric character (0-9)
            if len(value) == 1 and value.isdigit():
                return True  # It's a single numeric character
            else:
                return False  # It's a string, but not a single character
    return False  # It's not a single value

#%%
MIN_V = 1 # min week number possible
MIN_m = 1 # min month number possible
MIN_d = 1 # min day number possible
MIN_H = 0 # min hour number possible
MIN_M = 0 # min minute number possible

MAX_V = 53 # max week number possible
MAX_m = 12 # max month number possible
MAX_d = 31 # max day number possible
MAX_H = 23 # max hour number possible
MAX_M = 59 # max minute number possible

def _filter_dataframe_by_datetime(df, date_format, date_range):
    """
    Filter a DataFrame based on a specified date format and date range.

    Parameters:
    -----------
        df : pandas.DataFrame
            The DataFrame to be filtered.
        date_format :str
            The format of the date to be extracted from the DataFrame's index.
        date_range : int, str, list
            The range of dates to filter the DataFrame by.

    Returns:
    --------
        pandas.DataFrame: The filtered DataFrame.

    Raises:
    -------
        ValueError: If the date_format is not allowed or if the date_range is not a single value or a list.

    Note:
    -----
        The date_format parameter can be one of the following:
        - "%V": Week number
        - "%Y": Year
        - "%m-%d": Month and day
        - "%a": Weekday
        - "%m": Month
        - "%d": Day
        - "%y-%m-%d": Year, Month, and Day
        - "%H:%M": Hour and minute (e.g., "13:42")

        The date_range parameter can be one of the following:
        - A single value (int or str) representing a specific date or time.
        - A list of values (int or str) representing a range of dates or times.
        For example, ["01-01", "01-31"] represents a range of dates from January 1st to January 31st.

    """
    # Extract the date portion from the DatetimeIndex based on the specified date format
    try:
        df['date_type'] = df.index.strftime(date_format)
    except :
        raise ValueError('Error in the date_format. Check the date_format parameter')

    if _is_single_value(date_range) == True :
        if (date_format == "%V") or (date_format == "%Y") or (date_format == "%m") or (date_format == "%d"):
            # Filter by week number or by year or by month
            df_filtered = df[df['date_type'].astype(int) == date_range]
        elif (date_format == "%m-%d") or (date_format == "%H:%M") or (date_format == "%a") or (date_format == "%y-%m-%d"):
            # Filter by time (hour:minutes)
            # Filter by weekday (e.g., "Mon" for Monday)
            # Filter by year-month-day (e.g., "23-01-01" for January 1, 2023)
            # Filter by month and day (e.g., "01-01" for January 1)
            df_filtered = df[df['date_type'] == date_range]
        else :
            print("Filtering was no possible. date_format {} is not allowed".format(date_format))
            df_filtered = df

    elif (len(date_range)==1) or(len(date_range)>2):
        if (date_format == "%V") or (date_format == "%Y") or (date_format == "%m") or (date_format == "%d"):
            # Filter by week number or by year or by month
            df_filtered = df[df['date_type'].astype(int).isin(date_range)]
        elif (date_format == "%m-%d") or (date_format == "%H:%M") or (date_format == "%a") or (date_format == "%y-%m-%d"):
            # Filter by time (hour:minutes)
            # Filter by weekday (e.g., "Mon" for Monday)
            # Filter by year-month-day (e.g., "23-01-01" for January 1, 2023)
            # Filter by month and day (e.g., "01-01" for January 1)
            df_filtered = df[df['date_type'].isin(date_range)]
        else :
            print("Filtering was no possible. date_format {} is not allowed".format(date_format))
            df_filtered = df
    else:
        if isinstance(date_range, list) : 
            if (date_format == "%V") or (date_format == "%Y") or (date_format == "%m") or (date_format == "%d"):
                # Filter by week number or by year or by month
                df_filtered = df[df['date_type'].astype(int).between(date_range[0], date_range[1])]
            elif (date_format == "%m-%d") or (date_format == "%H:%M") or (date_format == "%a") or (date_format == "%y-%m-%d"):
                # Filter by time (hour:minutes)
                # Filter by weekday (e.g., "Mon" for Monday)
                # Filter by year-month-day (e.g., "23-01-01" for January 1, 2023)
                # Filter by month and day (e.g., "01-01" for January 1)
                if date_range[0] <= date_range[1] :
                    df_filtered = df[df['date_type'].between(str(date_range[0]), str(date_range[1]))]
                else :
                    if date_format == "%H:%M" : 
                        MAX = f'{MAX_H:02}:{MAX_M:02}'
                        df_filtered_part1 = df[df['date_type'].between(str(date_range[0]), MAX)] 
                        MIN = f'{MIN_H:02}:{MIN_M:02}'
                        df_filtered_part2 = df[df['date_type'].between(MIN, str(date_range[1]))]  
                        df_filtered = pd.concat([df_filtered_part1, df_filtered_part2], axis = 0)
                    else :
                        print("Filtering was no possible. The range must increase")
                        df_filtered = df
            else :
                print("Filtering was no possible. date_format {} is not allowed".format(date_format))
                df_filtered = df
        else :
            print("Filtering was no possible. date_range must be a single value or a list")
            df_filtered = df

    # Check if 'data_type' is in the DataFrame's columns
    if 'date_type' in df_filtered.columns:
        # If it exists, drop the column
        df_filtered = df_filtered.drop('date_type', axis=1)

    return df_filtered

#%%
def plot_shape(shape, params, row=0, ax=None, display_values=False):
    """ 
    Plot shape features in a bidimensional plot.
    
    Parameters 
    ---------- 
    shape: 1D array, pd.Series or pd.DataFrame 
        Shape features computed with shape_features function. 
    
    params: pd.DataFrame 
        Pandas dataframe returned by maad.features_rois.shape_features 
    
    row: int 
        Observation to be visualized 
    
    display_values: bool 
        Set to True to display the coefficient values. Default is False. 
    
    Returns 
    ------- 
    ax: matplotlib.axes 
        Axes of the figure 
    
    Examples 
    -------- 
    >>> from maad.sound import load, spectrogram 
    >>> from maad.features import shape_features, plot_shape 
    >>> import numpy as np 
    >>> s, fs = load('../data/spinetail.wav') 
    >>> Sxx, ts, f, ext = spectrogram(s, fs) 
    >>> shape, params = shape_features(np.log10(Sxx), resolution='high') 
    >>> plot_shape(shape, params) 

    """

    # compute shape of matrix
    dirs_size = params.theta.unique().shape[0]
    scale_size = np.unique(params.freq).size * np.unique(params.pyr_level).size
    # reshape feature vector
    idx = params.sort_values(["theta", "pyr_level", "scale"]).index

    if isinstance(shape, pd.DataFrame):
        shape_plt = shape.iloc[:, shape.columns.str.startswith("shp")]
        shape_plt = np.reshape(shape_plt.iloc[row, idx].values, (dirs_size, scale_size))
    elif isinstance(shape, pd.Series):
        shape_plt = shape.iloc[shape.index.str.startswith("shp")]
        shape_plt = np.reshape(shape_plt.iloc[idx].values, (dirs_size, scale_size))
    elif isinstance(shape, np.ndarray):
        shape_plt = np.reshape(shape_plt[idx], (dirs_size, scale_size))

    unique_scale = params.scale[idx] * 2 ** params.pyr_level[idx]
    # get textlab
    textlab = shape_plt
    textlab = np.round(textlab, 2)

    # plot figure
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    else:
        pass
    ax.imshow(
        shape_plt, aspect="auto", origin="lower", interpolation="None", cmap="viridis"
    )
    if display_values:
        for (j, i), label in np.ndenumerate(textlab):
            ax.text(i, j, label, ha="center", va="center")

    yticklab = np.round(params.theta.unique(), 1)
    xticklab = np.reshape(unique_scale.values, (dirs_size, scale_size))
    ax.set_xticks(np.arange(scale_size))
    ax.set_xticklabels(np.round(xticklab, 1)[0, :])
    ax.set_yticks(np.arange(dirs_size))
    ax.set_yticklabels(yticklab)
    ax.set_xlabel("Scale")
    ax.set_ylabel("Theta")
    plt.show()

    return ax

#%%
def overlay_centroid(im_ref, centroid, savefig=None, **kwargs):
    """ 
    Overlay centroids on the original spectrogram 
    
    Parameters 
    ---------- 
    Sxx :  2D array 
        Spectrogram 
        
    centroid: pandas DataFrame 
        DataFrame with centroid descriptors (centroid_f, centroid_t) 
        Do format_features(rois,tn,fn) before using overlay_centroid to be sure that 
        the format of the DataFrame is correct 
            
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
        
    kwargs, optional. This parameter is used by plt.plot and savefig functions 
            
        - savefilename : str, optional, default :'_spectro_overlaycentroid.png' 
            Postfix of the figure filename 
        
        - extent : list of scalars [left, right, bottom, top], optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        
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
        
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
        
        - format : string, optional, default is 'png' 
            Format to save the figure  
        
        ... and more, see matplotlib  

    Returns 
    ------- 
    ax  
        axis object (see matplotlib) 
    fig  
        figure object (see matplotlib) 

    Examples
    --------

    Get centroid from the whole power spectrogram 

    >>> from maad.sound import load, spectrogram
    >>> from maad.features import centroid_features
    >>> from maad.util import (power2dB, format_features, overlay_rois, plot2d,
                            overlay_centroid)
    
    Load audio and compute spectrogram 
    
    >>> s, fs = load('../data/spinetail.wav') 
    >>> Sxx,tn,fn,ext = spectrogram(s, fs, db_range=80) 
    >>> Sxx = power2dB(Sxx, db_range=80)
    
    Load annotations and plot
    
    >>> from maad.util import read_audacity_annot
    >>> rois = read_audacity_annot('../data/spinetail.txt') 
    >>> rois = format_features(rois, tn, fn) 
    >>> ax, fig = plot2d (Sxx, extent=ext)
    >>> ax, fig = overlay_rois(Sxx,rois, extent=ext, ax=ax, fig=fig)
    
    Compute the centroid of each rois, format to get results in the 
    temporal and spectral domain and overlay the centroids.
    
    >>> centroid = centroid_features(Sxx, rois) 
    >>> centroid = format_features(centroid, tn, fn)
    >>> ax, fig = overlay_centroid(Sxx,centroid, extent=ext, ax=ax, fig=fig)
    
    """
    
    # Check format of the input data
    if type(centroid) is not pd.core.frame.DataFrame:
        raise TypeError("Rois must be of type pandas DataFrame")

    if not (("centroid_t" and "centroid_f") in centroid):
        raise TypeError(
            "Array must be a Pandas DataFrame with column names:(centroid_t, centroid_f). Check example in documentation."
        )

    ylabel = kwargs.pop("ylabel", "Frequency [Hz]")
    xlabel = kwargs.pop("xlabel", "Time [s]")
    title = kwargs.pop("title", "ROIs Overlay")
    cmap = kwargs.pop("cmap", "gray")
    extent = kwargs.pop("ext", None)

    if extent is None:
        ylabel = "pseudofrequency [points]"
        xlabel = "pseudotime [points]"

    vmin = kwargs.pop("vmin", 0)
    vmax = kwargs.pop("vmax", 1)
    ax = kwargs.pop("ax", None)
    fig = kwargs.pop("fig", None)
    color = kwargs.pop("color", "firebrick")
    ms = kwargs.pop("ms", 2)
    marker = kwargs.pop("marker", "o")

    if (ax is None) and (fig is None):
        ax, fig = plot2d(
                        im_ref,
                        extent  = extent,
                        now     = False,
                        title   = title,
                        ylabel  = ylabel,
                        xlabel  = xlabel,
                        vmin    = vmin,
                        vmax    = vmax,
                        cmap    = cmap,
                        **kwargs
                        )

    ax.plot(centroid.centroid_t, centroid.centroid_f, marker, ms=ms, color=color)

    fig.canvas.draw()

    # SAVE FIGURE
    if savefig is not None:
        dpi = kwargs.pop("dpi", 96)
        format = kwargs.pop("format", "png")
        filename = kwargs.pop("filename", "_spectro_overlaycentroid")
        filename = savefig + filename + "." + format
        fig.savefig(filename, bbox_inches="tight", dpi=dpi, format=format, **kwargs)

    return ax, fig

#%%
def overlay_rois(im_ref, rois, edge_color=None, unique_labels= None, 
                 textbox_label=False, savefig=None, **kwargs):
    """ 
    Display bounding boxes with time-frequency regions of interest over a spectrogram.
    
    Regions of interest (ROIs) must be provided. They can be loaded as manual annotations or computed using automated methods (see the example section).
    
    Parameters 
    ---------- 
    im_ref : 2d ndarray of scalars 
        Spectrogram (or image) 

    rois_bbox : pandas.DataFrame
        Contains the bounding box of each ROI. The pandas.DataFrame must have columns
        ['min_x', 'max_x', 'min_y', 'max_y']. If your data is in the time-frequency
        format ['min_t', 'max_t', 'min_f', 'max_f'], use the function
        maad.util.format_features to get the cooresponding
        x, y coordinates.
        
    edge_color : {None, string, tuple, list of tuples}, default is None
        Define the color of the bounding-box to display.
        if it's a string, select a color name 
        if it's a tuple, give a tuple of floats [0,1] in the form (r,g,b,a), 
        with r for red, g for green, b for blue and a for alpha (i.e transparency)
        for example
        - pure red : (1,0,0,0)
        - pure yellow : (1,1,0,0)
        - pure blue with transparency(0,0,1,0.5)
        if it's a list, give a list of tuple or string. The number of element 
        in the list should be equal to the number of elements in unique_labels.
        if it's set to None, a random list of colors is used.
        
    unique_labels : list, default is None
        list of unique labels that could be either strings or numbers. When
        a list is given as argument, the number element in the list should 
        be the same as the number of edge_color
    
    textbox_label: bool
        Display a text box above the bounding box indicating the name of the label.
        The rois_bbox pandas.DataFrame must have a column called 'label' with names
        assigned to each ROI.

    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
        
    kwargs, optional. This parameter is used by plt.plot and savefig functions 
            
        - savefilename : str, optional, default :'_spectro_overlayrois.png' 
            Postfix of the figure filename 
            
        - extent : list of scalars [left, right, bottom, top], optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.  
        
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
        
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
        
        - format : string, optional, default is 'png' 
            Format to save the figure  
        
        ... and more, see matplotlib  

    Returns 
    ------- 
    ax : axis object (see matplotlib) 
        
    fig : figure object (see matplotlib) 
    
    Examples 
    -------- 
    
    Simple display of bounding boxes over the spectrogram.
    
    >>> from maad import sound, util
    >>> s, fs = sound.load('./data/spinetail.wav')
    >>> rois = util.read_audacity_annot('./data/spinetail.txt')
    >>> Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=512, noverlap=256)
    >>> rois = util.format_features(rois, tn, fn)
    >>> fig, ax = plt.subplots(figsize=(10,5))
    >>> util.plot_spectrogram(Sxx,ext, db_range=70, ax=ax)
    >>> util.overlay_rois(Sxx, rois, fig=fig, ax=ax, textbox_label=True)
    
    Detect regions of interest and display them over the spectrogram.
    Load audio recording and compute the spectrogram.
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, fcrop=(0,10000))   

    Subtract the background noise before finding ROIs.
    
    >>> Sxx_noNoise = maad.sound.median_equalizer(Sxx)
    
    Convert linear spectrogram into dB and smooth.
    
    >>> Sxx_noNoise_dB = maad.util.power2dB(Sxx_noNoise)         
    >>> Sxx_noNoise_dB_blurred = maad.sound.smooth(Sxx_noNoise_dB)
    
    Detection of the acoustic signature => creation of a mask
    
    >>> im_bin = maad.rois.create_mask(Sxx_noNoise_dB_blurred, bin_std=6, bin_per=0.5, mode='relative') 
    
    Select rois from the mask and display bounding box over the spectrogram without noise.
    
    >>> import numpy as np
    >>> im_rois, df_rois = maad.rois.select_rois(im_bin, min_roi=100)  
    >>> maad.util.overlay_rois (Sxx_noNoise_dB, df_rois, extent=ext,vmin=np.median(Sxx_noNoise_dB), vmax=np.median(Sxx_noNoise_dB)+60) 
        
    """

    # Check format of the input data
    if type(rois) is not pd.core.frame.DataFrame:
        raise TypeError("Rois must be of type pandas DataFrame.")
    if not (("min_y" and "min_x" and "max_y" and "max_x") in rois):
        raise TypeError(
            "Array must be a Pandas DataFrame with column names: min_y, min_x, max_y, max_x. Check example in documentation."
        )
        
    ylabel = kwargs.pop("ylabel", "Frequency [Hz]")
    xlabel = kwargs.pop("xlabel", "Time [s]")
    title = kwargs.pop("title", "ROIs Overlay")
    cmap = kwargs.pop("cmap", "gray")
    vmin = kwargs.pop("vmin", np.percentile(im_ref, 0.05))
    vmax = kwargs.pop("vmax", np.percentile(im_ref, 0.95))
    ax = kwargs.pop("ax", None)
    fig = kwargs.pop("fig", None)
    extent = kwargs.pop("extent", None)

    if extent is None:
        xlabel = "pseudoTime [points]"
        ylabel = "pseudoFrequency [points]"
        
    if (ax is None) and (fig is None):
        ax, fig = plot2d(
                        im_ref,
                        extent  = extent,
                        now     = False,
                        title   = title,
                        ylabel  = ylabel,
                        xlabel  = xlabel,
                        vmin    = vmin,
                        vmax    = vmax,
                        cmap    = cmap,
                        **kwargs
                        )

    # Convert pixels into time and frequency values
    y_len, x_len = im_ref.shape
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_scaling = (xmax - xmin) / x_len
    y_scaling = (ymax - ymin) / y_len

    # if no list of unique labels is provided, try to find a "label" column
    if unique_labels is None :
        # test if rois has a label column
        if "label" in rois:
            # select the label column
            rois_label = rois.label.values
            uniqueLabels = list(np.unique(np.array(rois_label)))
        else:
            uniqueLabels = [0]    
    else: 
        if isinstance(unique_labels,list) :     
            uniqueLabels = unique_labels
        elif isinstance(unique_labels,np.ndarray) :     
            uniqueLabels = list(unique_labels)
        elif isinstance(unique_labels, str) :
            uniqueLabels = [unique_labels]
        else:
            # test if unique_labels is a number
            try:
                int(unique_labels)
                uniqueLabels = [unique_labels]
            except:
                raise TypeError(
                    "unique_labels must be a list or a string or a number"
                )      
        
    # Colormap
    if edge_color is None :
        # in order to still have the color yellow by default in case of no label
        # or a unique label
        if len(uniqueLabels) == 1:
            color = tuple(["yellow"])
        else:
            color_func = rand_cmap(len(uniqueLabels) + 1, first_color_black=False)
            color = color_func(np.arange(len(uniqueLabels)))
    # if a list of colors is defined, test if the length of the list is 
    # enough compared to the number of unique labels
    # convert into tuple
    elif isinstance(edge_color, str) :    
        color = tuple([edge_color]) * len(uniqueLabels)
    elif isinstance(edge_color, list) :  
        nn = int(len(uniqueLabels)/len(edge_color)) +1
        color = tuple(edge_color) * nn

    cc = 0
    for index, row in rois.iterrows():
        cc = cc + 1
        y0 = row["min_y"]
        x0 = row["min_x"]
        y1 = row["max_y"]
        x1 = row["max_x"]
        # find the position of the label in uniqueLabels
        if "label" in row :
            ii = uniqueLabels.index(row["label"])
        else:
            ii = 0
        rect = mpatches.Rectangle(
            (x0 * x_scaling + xmin, y0 * y_scaling + ymin),
            (x1 - x0) * x_scaling,
            (y1 - y0) * y_scaling,
            fill=False,
            edgecolor=color[ii],
            linewidth=1,
        )
        # draw the rectangle
        ax.add_patch(rect)
        
        if textbox_label:
            textbox = dict(boxstyle='square', fc=color[ii], 
                        ec=color[ii], alpha=1, pad=0, linewidth=2)
            ax.text(x0*x_scaling + xmin, (y1+2)*y_scaling + ymin, 
                    str(row.label), 
                    color='white',
                    fontweight='semibold',
                    fontsize='small',
                    fontfamily='sans-serif',
                    fontvariant='small-caps',
                    bbox=textbox
                    )
            
    if fig is not None:
        fig.canvas.draw()

    # SAVE FIGURE
    if savefig is not None and fig is not None:
        dpi = kwargs.pop("dpi", 96)
        format = kwargs.pop("format", "png")
        filename = kwargs.pop("filename", "_spectro_overlayrois")
        filename = savefig + filename + "." + format
        fig.savefig(filename, bbox_inches="tight", dpi=dpi, format=format, **kwargs)

    return ax, fig

#%%
def plot1d(x, y, ax=None, **kwargs):
    """
    Plot the waveform or spectrum of an audio signal. 
    
    Parameters
    ----------
    x : 1d ndarray of integer
        Vector containing the abscissa values (horizontal axis)
        
    y : 1d ndarray of scalar
        Vector containing the ordinate values (vertical axis)  
    
    ax : axis, optional, default is None
        Draw the signal on this specific axis. Allow multiple plots on the same
        axis.
            
    kwargs, optional
        - figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        - facecolor : matplotlib color, optional, default: 'w' (white)
            the background color.  
        - edgecolor : matplotlib color, optional, default: 'k' (black)
            the border color. 
        - color : matplotlib color, optional, default: 'k' (black)
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
            In addition, you can specify colors in many ways, including RGB 
            tuples (0.2,1,0.5). See matplotlib color
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
    >>> import numpy as np    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    
    Plot the audiogram
    
    >>> tn = np.arange(0,len(s))/fs
    >>> fig, ax = maad.util.plot1d(tn,s)
    
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(s,fs)
    
    Convert spectrogram into dB SPL
    
    >>> Lxx = maad.spl.power2dBSPL(Sxx_power, gain=42) 
    
    Plot the spectrum at t = 7s
    
    >>> index = maad.util.nearest_idx(tn,7)
    >>> fig_kwargs = {'figtitle':'Spectrum (PSD)',
                    'xlabel':'Frequency [Hz]',
                    'ylabel':'Power [dB]',
                    'linewidth': 0.5
                    }
    
    >>> fig, ax = maad.util.plot1d(fn, Lxx[:,index], **fig_kwargs)
    """

    figsize = kwargs.pop("figsize", (4, 10))
    facecolor = kwargs.pop("facecolor", "w")
    edgecolor = kwargs.pop("edgecolor", "k")
    linewidth = kwargs.pop("linewidth", 0.5)
    color = kwargs.pop("color", "k")
    title = kwargs.pop("figtitle", "")
    xlabel = kwargs.pop("xlabel", "Time [s]")
    ylabel = kwargs.pop("ylabel", "Amplitude [AU]")
    legend = kwargs.pop("legend", None)
    now = kwargs.pop("now", False)

    # if no ax, create a figure and a subplot associated a figure otherwise
    # find the figure that belongs to ax
    if ax is None:
        fig, ax = plt.subplots()
        # set the paramters of the figure
        fig.set_figheight(figsize[0])
        fig.set_figwidth(figsize[1])
        fig.set_facecolor(facecolor)
        fig.set_edgecolor(edgecolor)
    else:
        fig = ax.get_figure()

    # plot the data on the subplot
    line = ax.plot(x, y, color=color, linewidth=linewidth, **kwargs)

    # set legend to the line
    line[0].set_label(legend)

    # set the parameters of the subplot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.axis("tight")
    ax.grid(True, linewidth=0.3)
    ax.margins(x=0)
    
    if legend is not None:
        ax.legend()

    # Display the figure now
    if now:
        plt.show()

    return ax, fig

#%%
def plot_wave(s, fs, tlims=None, ax=None, **kwargs):
    """
    Plot audio waveform.
    
    Parameters
    ----------
    s : 1d ndarray
        Audio signal.
    fs : int
        Sampling rate of audio signal.
    tlims : tuple, optional
        Minimum and maximum temporal limits for the display (min_t, max_t). 
        The default is None.
    ax : matplotlib.axes, optional
        Pre-existing axes for the plot. The default is None.
    **kwargs : matplotlib figure properties
        Other keyword arguments that are passed down to matplotlib.axes.

    Returns
    -------
    ax : matplotlib.axes
        The matplotlib axes associated to plot.
    
    See also
    --------
    maad.sound.plot_spectrum
        
    Examples
    --------
    
    Plot a waveform of an audio signal.
    >>> from maad import sound
    >>> from maad.util import plot_wave
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> ax = plot_wave(s, fs)
    
    Use `plot_wave` with predifined matplotlib axes.
    >>> import matplotlib.pyplot as plt
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> fig, ax = plt.subplots(2,1)
    >>> plot_wave(s, fs, ax=ax[0], xlabel='', figtitle='Spinetail')
    >>> plot_wave(s, fs, tlims=(5,8), ax=ax[1])
    """

    if tlims is not None:
        s = s[int(tlims[0] * fs) : int(tlims[1] * fs)]
        t = np.linspace(0, s.shape[0] / fs, s.shape[0])
        t = t + tlims[0]  # add minimum value
    else:
        t = np.linspace(0, s.shape[0] / fs, s.shape[0])

    ax, fig = plot1d(t, s, ax, **kwargs)

    return ax

#%%
def plot_spectrum(pxx, f_idx, ax=None, flims=None, log_scale=False, fill=True, **kwargs):
    """
    Plot power spectral density estimate (PSD).
    
    Parameters
    ----------
    pxx : 1d ndarray
        Power spectral density estimate computed with `maad.sound.spectrum`.
        
    f_idx : 1d ndarray
        Index of frequencies associated with the PSD.
    
    ax : matplotlib.axes, optional
        Pre-existing axes for the plot. The default is None.
        
    flims : tuple, optional
        Minimum and maximum spectral limits for the display (min_f, max_f). 
        Default is None.

    log_scale : bool, optional
        Use a logarithmic scale to display amplitude values. The default is False.
    
    fill : bool, optional
        Fill the area between the curve and the minimum value.
        
    **kwargs : matplotlib figure properties
            Other keyword arguments that are passed down to matplotlib.axes.

    Returns
    -------
    ax : matplotlib.axes
        The matplotlib axes associated to plot.

    See also
    --------
    maad.sound.spectrum, maad.util.plot_wave
    
    Examples
    --------
    
    Plot a spectrum of an audio signal.
    
    >>> from maad import sound, util
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> pxx, f_idx = sound.spectrum(s, fs, nperseg=1024)
    >>> util.plot_spectrum(pxx, f_idx)
    
    Use `plot_spectrum` with predifined matplotlib axes.
    
    >>> import matplotlib.pyplot as plt
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> s_slice = sound.trim(s, fs, 5, 8)
    >>> pxx, f_idx = sound.spectrum(s_slice, fs, nperseg=1024)
    >>> fig, ax = plt.subplots(2,1, figsize=(10,6))
    >>> util.plot_wave(s_slice, fs, ax=ax[0])
    >>> util.plot_spectrum(pxx, f_idx, ax=ax[1], log_scale=True)
    
    """
    ylabel = kwargs.pop("ylabel", "Amplitude [AU]")
    xlabel = kwargs.pop("xlabel", "Frequency [Hz]")

    if log_scale == True:
        pxx = power2dB(pxx)
        ylabel = 'Amplitude [dB]'

    if flims is not None:
        pxx = pxx[(f_idx >= flims[0]) & (f_idx <= flims[1])]
        f_idx = f_idx[(f_idx >= flims[0]) & (f_idx <= flims[1])]

    amp_min = pxx.min()
    ax, fig = plot1d(f_idx, pxx, ax=ax, ylabel=ylabel, xlabel=xlabel, **kwargs)
    if fill:
        ax.fill_between(f_idx, amp_min, pxx, alpha=0.3, fc="grey")

    return ax

#%%
def plot2d(im, ax=None, colorbar=True, **kwargs):
    """
    Display the spectrogram of an audio signal. 
    
    The spectrogram should be previously computed using the function 
    ``maad.sound.spectrogram``.
    
    Parameters
    ----------
    im : 2D numpy array 
        Image or Spectrogram
        
    ax : axis, optional, default is None
        Draw the image on this specific axis. Allow multiple plots on the same
        figure.
            
    kwargs, optional
        - figsize : tuple of integers, optional, default: (4,13)
            width, height in inches.  
        - title : string, optional, default : 'Spectrogram'
            title of the figure
        - xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        - ylabel : string, optional, default : 'Frequency [Hz]'
            label of the vertical axis
        - xticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.
        - yticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.
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
        - interpolation : list of string [None, 'none', 'nearest', 'bilinear', 
            'bicubic', 'spline16','spline36', 'hanning', 'hamming', 'hermite', 
            'kaiser', 'quadric','catrom', 'gaussian', 'bessel', 'mitchell', 
            'sinc', 'lanczos'], optional, default : None
            
        ... and more, see matplotlib
        
    Returns
    -------
    fig : Figure
        The Figure instance 
    ax : Axis
        The Axis instance
        
    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(w,fs)
    >>> Lxx = maad.spl.power2dBSPL(Sxx_power, gain=42) # convert into dB SPL
    >>> fig_kwargs = {'vmax': Lxx.max(),
                    'vmin':0,
                    'extent':(tn[0], tn[-1], fn[0], fn[-1]),
                    'title':'Power spectrogram density (PSD) in dB SPL',
                    'xlabel':'Time [s]',
                    'ylabel':'Frequency [Hz]',
                    }
    >>> ax, fig = maad.util.plot2d(Lxx,interpolation=None,**fig_kwargs)      
        
    """

    # matplotlib parameters
    title = kwargs.pop("title", "")
    ylabel = kwargs.pop("ylabel", "Frequency [Hz]")
    xlabel = kwargs.pop("xlabel", "Time [s]")
    xticks = kwargs.pop("xticks", None)
    yticks = kwargs.pop("yticks", None)
    cmap = kwargs.pop("cmap", "gray")
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    extent = kwargs.pop("extent", None)
    now = kwargs.pop("now", False)

    if extent is not None:
        figsize = kwargs.pop(
            "figsize",
            (0.20 * (extent[3] - extent[2]) / 1000, 0.33 * (extent[1] - extent[0])),
        )
    else:
        figsize = kwargs.pop("figsize", (4, 13))

    # if no ax, create a figure and a subplot associated a figure otherwise
    # find the figure that belongs to ax
    if ax is None:
        fig, ax = plt.subplots()
        # set the paramters of the figure
        fig.set_facecolor("w")
        fig.set_edgecolor("k")
        fig.set_figheight(figsize[0] + 1)
        fig.set_figwidth(figsize[1])
    else:
        fig = ax.get_figure()

    # display image
    _im = ax.imshow(im, extent=extent, origin="lower", cmap=cmap, 
                    vmin=vmin, vmax=vmax, **kwargs)
    if colorbar==True:
        plt.colorbar(_im, ax=ax)

    # set the parameters of the subplot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(ticks=xticks[0])
        if len(xticks) == 2:
            ax.set_xticklabels(labels=xticks[1])
    if yticks is not None:
        ax.set_yticks(ticks=yticks[0])
        if len(yticks) == 2:
            ax.set_yticklabels(labels=yticks[1])
    ax.axis("tight")

    fig.tight_layout()

    # Display the figure now
    if now:
        plt.show()

    return ax, fig

#%%
def plot_spectrogram(Sxx, extent, db_range=96, gain=0, log_scale=True,
                     colorbar=True, interpolation='bilinear', **kwargs):
    """
    Plot spectrogram represenation.
    
    Parameters
    ----------
    Sxx : 2d ndarray
        Spectrogram computed using `maad.sound.spectrogram`.
    extent : list of scalars [left, right, bottom, top]
        Location, in data-coordinates, of the lower-left and upper-right corners.
    db_range : int or float, optional
        Range of values to display the spectrogram. The default is 96.
    gain : int or float, optional
        Gain in decibels added to the signal for display. The default is 0.
    log_scale : bool, optional
        Logarithmic transformation of spectrogram coefficients to enhace visual 
        representation. The default is True.
    colorbar : bool, optional
        Plot the colorbar next to the spectrogram. The default is True.
    interpolation : bool, optional
        Interpolate spectrogram values to enhance visual representation. 
        The default is 'bilinear'.
    **kwargs : matplotlib figure properties
            Other keyword arguments that are passed down to matplotlib.axes.

    Returns
    -------
    ax : matplotlib.axes
        The matplotlib axes associated to plot.

    See also
    --------
    maad.sound.plot_spectrum, maad.util.plot_wave
    
    Examples
    --------
    
    Plot the spectrogram of an audio.
    
    >>> from maad import sound, util
    >>> s, fs = sound.load('../data/spinetail.wav') 
    >>> Sxx, tn, fn, ext = sound.spectrogram(s,fs)
    >>> util.plot_spectrogram(Sxx, ext, db_range=50, gain=30, figsize=(4,10))
        
    Use `plot_spectrogram` with predifined matplotlib axes.
    
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(2,1, figsize=(10,6))
    >>> util.plot_wave(s, fs, ax=ax[0])
    >>> util.plot_spectrogram(Sxx, ext, db_range=50, gain=30, colorbar=False, ax=ax[1])

    Plot a single frase of the spinetail.
    
    >>> Sxx, tn, fn, ext = sound.spectrogram(s,fs, flims=(2000,15000), tlims=(5,8))
    >>> util.plot_spectrogram(Sxx, ext, db_range=50, gain=30, figsize=(4,6))
    """
    if log_scale:
        Sxx_db = power2dB(Sxx, db_range, gain)
    else :
        Sxx_db = Sxx
    
    ax, fig = plot2d(Sxx_db, extent=extent, colorbar=colorbar, 
                     interpolation=interpolation, **kwargs)
    return ax

#%%
def rand_cmap(
    nlabels,
    type="bright",
    first_color_black=True,
    last_color_black=False,
    seed=321,
    verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    
    Parameters
    ----------
    nlabels : int
        Number of labels (size of colormap)
    type : string
        'bright' for strong colors, 'soft' for pastel colors. Default is 'bright'
    first_color_black : bool, optional
        Option to use first color as black. Default is True  
    last_color_black : bool, optional   
        Option to use last color as black. Default is False
    seed : int, optional
        Fix the seed of the random engine. Default is 321   
    verbose  : bool, optional   
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
    np.random.seed(seed=seed)

    if verbose:
        print("Number of labels: " + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == "bright":
        randHSVcolors = [
            (
                np.random.uniform(low=0.1, high=1),
                np.random.uniform(low=0.2, high=1),
                np.random.uniform(low=0.5, high=1),
            )
            for i in range(nlabels)
        ]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(
                colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])
            )

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == "soft":
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
            )
            for i in range(nlabels)
        ]

    if first_color_black:
        randRGBcolors[0] = [0, 0, 0]

    if last_color_black:
        randRGBcolors[-1] = [0, 0, 0]

    random_colormap = LinearSegmentedColormap.from_list(
        "new_map", randRGBcolors, N=nlabels
    )

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        colorbar.ColorbarBase(
            ax,
            cmap=random_colormap,
            norm=norm,
            spacing="proportional",
            ticks=None,
            boundaries=bounds,
            format="%1i",
            orientation=u"horizontal",
        )

    return random_colormap


#%%
def crop_image(im, tn, fn, fcrop=None, tcrop=None):
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
    
    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram(w,fs)
    >>> Lxx = maad.spl.power2dBSPL(Sxx_power, gain=42) # convert into dB SPL
    >>> fig_kwargs = {'vmax': Lxx.max(),
                    'vmin':0,
                    'extent':(tn[0], tn[-1], fn[0], fn[-1]),
                    'title':'Power spectrogram density (PSD)',
                    'xlabel':'Time [s]',
                    'ylabel':'Frequency [Hz]',
                    }
    >>> fig, ax = maad.util.plot2d(Lxx,**fig_kwargs)      
    >>> Lxx_crop, tn_crop, fn_crop = maad.util.crop_image(Lxx, tn, fn, fcrop=(2000,10000), tcrop=(0,30))
    >>> fig_kwargs = {'vmax': Lxx.max(),
                    'vmin':0,
                    'extent':(tn_crop[0], tn_crop[-1], fn_crop[0], fn_crop[-1]),
                    'title':'Crop of the power spectrogram density (PSD)',
                    'xlabel':'Time [s]',
                    'ylabel':'Frequency [Hz]',
                    }
    >>> fig, ax = maad.util.plot2d(Lxx_crop,**fig_kwargs)  
    """

    if tcrop is not None:
        indt = (tn >= tcrop[0]) * (tn <= tcrop[1])
        im = im[:, indt]
        # redefine tn
        tn = tn[np.where(indt > 0)]
    if fcrop is not None:
        indf = (fn >= fcrop[0]) * (fn <= fcrop[1])
        im = im[
            indf,
        ]
        # redefine fn
        fn = fn[np.where(indf > 0)]

    return im, tn, fn


#%%
def save_figlist(fname, figlist):
    """
    Save a list of figures or spectrograms to disk.
    
    Parameters
    ----------
    fname: string
        Suffix string add to the filename. The extension should be specified since it 
        indicates the image format.

    Notes
    -----
    This function does not return any variable.
        
    """
    for i, fig in enumerate(figlist):
        fname_save = "%d_%s" % (i, fname)
        imsave(fname_save, fig)

#%%
def plot_features_map(df, norm=True, mode="24h", **kwargs):
    """
    Plot features values on a heatmap.
    
    The plot has the features the vertical axis and the time on the horizontal axis.
    
    Parameters
    ----------
    df : Panda DataFrame
        DataFrame with features (ie. indices).
    
    norm : boolean, default is True
        if True, the features are scaled between 0 to 1
        
    mode : string in {'24h'}, default is '24h'
        Select if the timeline of the phenology:
            
            -'24h' : average of the results over a day
            - otherwise, the timeline is the timeline of the dataframe
            
    **kwargs
        Specific to this function:
        
        - ftime : Time format to display as x label by default '%Y-%m-%d'(see https://docs.python.org/fr/3.6/library/datetime.html?highlight=strftime#strftime-strptime-behavior)
            
        Specific to matplotlib:
            
        - figsize : tuple of integers, optional, default: (4,10) width, height in inches.  
        
        - title : string, optional, default : 'Spectrogram' title of the figure
        
        - xlabel : string, optional, default : 'Time [s]' label of the horizontal axis

        - ylabel : string, optional, default : 'Amplitude [AU]' label of the vertical axis
            
        - xticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.
            
        - yticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.
        
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
    see plot_extract_alpha_indices.py advanced example for a complete example
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> np.random.seed(2021)
    >>> M = np.random.rand(24, 7)
    >>> df = pd.DataFrame(M)
    >>> indices = ['A','B','C','D','E','F','G']
    >>> df.columns = indices
    >>> df.index =pd.date_range(start=pd.Timestamp('00:00:00'), end=pd.Timestamp('23:00:00'), freq='1H')
    >>> maad.util.plot_features_map(df) 
    """

    if isinstance(df, pd.DataFrame) == False:
        raise TypeError("df must be a Pandas Dataframe")
    elif isinstance(df.index, pd.DatetimeIndex) == False:
        raise TypeError("df must have an index of type DateTimeIndex")

    if mode == "24h":
        # Mean values by hour
        df = df.groupby(df.index.hour).mean()
        # Get the list of unique index of type 'hour'
        x_label = [i + j for i, j in zip(map(str, df.index.values), ["h"] * len(df))]
    else:
        # Get the list of unique index of type 'hour'
        ftime = kwargs.pop("ftime", "%Y-%m-%d")
        x_value = df.index.strftime(ftime)
        x_label = x_value.tolist()

    if norm:
        df = linear_scale(df)

    # kwargs
    cmap = kwargs.pop("cmap", "RdBu_r")
    figsize = kwargs.pop("figsize", None)

    # plot
    if figsize is None :
        fig = plt.figure(figsize=(len(df)*0.33, len(list(df))*0.27))
    ax = fig.add_subplot(111)
    caxes = ax.matshow(df.transpose(), cmap=cmap, aspect="auto", **kwargs)
    if norm :
        fig.colorbar(caxes, shrink=0.75, label="Normalized value")
    else :
        fig.colorbar(caxes, shrink=0.75, label="Value")    
    # Set ticks on both sides of axes on
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    # We want to show all ticks...
    ax.set_yticks(np.arange(len(df.columns)))
    ax.set_xticks(np.arange(len(x_label)))
    # ... and label them with the respective list entries
    ax.set_yticklabels(df.columns)
    ax.set_xticklabels(x_label)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=10)
    fig.tight_layout()
    plt.show()

    return fig, ax

#%%
def heatmap_by_date_and_time (
                    dataframe,
                    disp_column,
                    date_format = "%V", # see https://docs.python.org/3/library/datetime.html 
                    date_range = [1,53],
                    time_resolution = "30T",
                    time_range = ["00:00", "23:59"],
                    start_hour = "00:00",
                    full_display = False,
                    date_min_to_disp = 1,
                    date_max_to_disp = 53,
                    cb_legend = "",
                    display = True,
                    verbose = False,
                    **kwargs,
                    ) :
    """
    Plot a heatmap of a features by time (x-axis) and date (y-axis).

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input DataFrame containing the data.
        Must contain a column (or index) date in the format %Y-%m-%d %H:%M%S
    disp_column : str
        The name of the column to be displayed.
    date_format : str, optional
        The format of the date. The default is "%V".
        (See https://docs.python.org/3/library/datetime.html for format codes)
        Possible format are :
        - "%V" for week number (from 1 to 53)
        - "%m" for Month (from 1 to 12)
        - "%d" for Day (from 1 to 31 depending on the month)
        - "%m-%d" for Date without year (from 01-01 to 12-31)
        - "%y-%m-%d" for Date with year 
    date_range : list of int, optional
        The range of date types to include. The default is [1, 53].
        The format of the range depends on date_format.
        For instance, to get all the samples in the March, date_range would be 
        [03-01, 03-31] and date_format would be "%m-%d"
    time_resolution : str, optional
        The time resolution. The default is "30T".
        T is for minute. 30T means a time resolution of 30 min. 
        Everything within this intervale is averaged
        Other time formats are H for hour and D for day.
    time_range : list of str, optional
        The time range to consider. The default is ["00:00", "23:59"].
    start_hour : str, optional
        The start hour. The default is "12:00".
    full_display : bool, optional
        Whether to display the full date range. The default is False.
    date_min_to_disp : int, optional
        The minimum date on the y-axis. The default is 1.
        The value depends on the date_format. 
        See date_format to know the possible formats and the range of possible values
    date_max_to_disp : int, optional
        The maximum date on the y-axis. The default is 53.
        The value depends on the date_format. 
        See date_format to know the possible formats and the range of possible values
    cb_legend : str, optional
        The colorbar legend label. The default is "".
    verbose : bool, optional
        If True, display verbose information. The default is False.
    **kwargs
        Specific to this function:
        
        - ftime : Time format to display as x label by default '%Y-%m-%d'(see https://docs.python.org/fr/3.6/library/datetime.html?highlight=strftime#strftime-strptime-behavior)
            
        Specific to matplotlib:
            
        - figsize : tuple of integers, optional, default: (4,10) width, height in inches.  
        
        - title : string, optional, default : 'Spectrogram' title of the figure
        
        - xlabel : string, optional, default : 'Time [s]' label of the horizontal axis

        - ylabel : string, optional, default : 'Amplitude [AU]' label of the vertical axis
            
        - xticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.
            
        - yticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.
        
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
    df_mean : pd.DataFrame
        The heatmap with mean values.
    
    df_std :pd.DataFrame
        The heatmap with standard deviation values.

    fig : matplotlib.figure.Figure
        The generated matplotlib figure.

    ax : matplotlib.axes.Axes
        The generated matplotlib axis.
    """

    # test if dataframe is not a pd.DataFrame
    if isinstance(dataframe, pd.DataFrame) == False:
        if verbose :
            print("***WARNING*** dataframe must be a valid pandas dataframe")
        return
    
    # copy the dataframe
    df = dataframe.copy()

    # keep some columns
    df = df.filter([disp_column,'date'])

    try :
        if not('date' in df)  :  
            df = df.reset_index(['date'])
        # convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        # set the index to date
        df.set_index(['date'], inplace=True)
        # convert index to datetime 
        df.index = pd.DatetimeIndex(df.index)
        # sort dataframe by date
        df = df.sort_index(axis=0)
    except :
        raise Exception("***WARNING*** df must have a valid date index that could be converted in Datetime type")

    # filter by time
    df = _filter_dataframe_by_datetime(df, "%H:%M", time_range)
    # filter by date_type
    df = _filter_dataframe_by_datetime(df, date_format, date_range)

    # # Extract hour and minute
    df['time'] = df.index.strftime("%H:%M")

    # # Extract week, year, weekday or date
    df['date_type'] = df.index.strftime(date_format) 

    # create a matrix with date_type (week, year, weekday, date) as columns and time (hour) as rows
    df_mean = df.groupby(['time', 'date_type'])[disp_column].aggregate('mean').unstack()

    # Resample the timeline. Need to reformat the time in datetime before resampling
    df_mean = df_mean.reset_index()
    df_mean.index = pd.to_datetime(df_mean['time'], format='%H:%M')
    df_mean.drop(columns=['time'], inplace = True)
    df_mean = df_mean.resample(time_resolution).mean()
    df_std = df_mean.resample(time_resolution).std()
    df_mean.index = df_mean.index.strftime("%H:%M")
    df_std.index = df_std.index.strftime("%H:%M")

    if display == True : 

        # try to use seaborn if installed
        try:
            import seaborn as sns
            sns.set_theme("paper")
        except ImportError as e:
            if verbose :
                print("seaborn is not installed, use default settings")

        if full_display == True :
            # Create a list of unique index of type 'hour' for x axis
            time_min = datetime.strptime("00:00", "%H:%M")
            time_max = datetime.strptime("23:59", "%H:%M")
            x_label = pd.date_range(time_min, time_max, freq=time_resolution).strftime("%H:%M")
            x_label = x_label.values
            # Create a vector for y axis depending on date_format
            if date_format == "%V" :
                y_label = [f'{x:02d}' for x in np.arange(date_min_to_disp,date_max_to_disp+1)]
            elif date_format == "%m" :
                y_label = [f'{x:02d}' for x in np.arange(date_min_to_disp,date_max_to_disp+1)]   
            elif (date_format == "%y-%m-%d") or (date_format == "%m-%d") or (date_format == "%d"):
                date_min_to_disp = datetime.strptime(str(date_min_to_disp), date_format)
                date_max_to_disp = datetime.strptime(str(date_max_to_disp), date_format)
                y_label = pd.date_range(date_min_to_disp, date_max_to_disp, freq='D').strftime(date_format)
                y_label = y_label.values
            else:
                # by default per week
                y_label = [f'{x:02d}' for x in np.arange(1,53)]
            # create an prefilled dataframe with all weeks and time
            df2 = pd.DataFrame(index=x_label, 
                            columns=y_label,
                            dtype = 'float')
            df2.reset_index(inplace=True)
            df2 = df2.rename(columns={"index":"time"})
            df2.set_index("time", inplace=True)

            for idx, row in df_mean.iterrows() :
                df2.loc[idx] = row
            
            df_mean=df2
        
        # rename the unique column with the corresponding name
        if date_format == "%V" : df_mean.columns.name='Week number'
        if date_format == "%m" : df_mean.columns.name='Month'
        if date_format == "%d" : df_mean.columns.name='Day'
        if date_format == "%m-%d" : df_mean.columns.name='Date'
        if date_format == "%y-%m-%d" : df_mean.columns.name='Date'

        # shift the time to start at start_hour and finish at start_hour, in order to center
        # the graph at start_hour + 12h
        df_part1 = df_mean[df_mean.index >= start_hour]
        df_part2 = df_mean[df_mean.index < start_hour]
        df_part2.sort_index(inplace=True)
        df = pd.concat([df_part1,df_part2])  
        
        # Get the list of unique index of type 'hour'
        x_label = [i + j for i, j in zip(map(str, df.index.values), [" "] * len(df))]

        # plot
        # matplotlib parameters
        fig = kwargs.pop("fig", None)
        ax = kwargs.pop("ax", None) 
        xlabel = kwargs.pop("xlabel", "Hours")
        ylabel = kwargs.pop("ylabel", df_mean.columns.name)
        now = kwargs.pop("now", True)
        vmin = kwargs.pop("vmin", np.nanpercentile(df,1))
        vmax = kwargs.pop("vmax", np.nanpercentile(df,99))
        cmap = kwargs.pop("cmap", "RdPu")
        title = kwargs.pop("title", None)
        figsize = kwargs.pop("figsize", (len(df)*0.33*0.75, len(list(df))*0.26*0.75 +0.75))
        
        # Create a figure and a subplot associated
        if (fig is None) or (ax is None) :
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        # display image
        caxes = ax.matshow(df.transpose(), 
                        interpolation = "None",
                        aspect="auto", 
                        cmap = cmap,
                        vmin = vmin,
                        vmax = vmax,
                        **kwargs
                        )
        if cb_legend =="" :
            fig.colorbar(caxes, shrink=0.75, label=disp_column)
        else:
            fig.colorbar(caxes, shrink=0.75, label=cb_legend)
        # Set ticks on both sides of axes on
        ax.tick_params(which='major', axis="x", bottom=False, top=False, 
                    labelbottom=True, labeltop=False)
        ax.tick_params(which='minor', axis="x", length = 0)
        # We want to show all ticks...
        ax.set_yticks(np.arange(len(df.columns)))
        ax.set_xticks(np.arange(len(x_label)))
        # ... and label them with the respective list entries
        ax.set_yticklabels(df.columns)
        ax.set_xticklabels(x_label)
        
        # Minor ticks
        ax.set_xticks(np.arange(-0.5, len(df.index), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(list(df)), 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='major', color='w', linestyle='-', linewidth=0)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        
        # add title to the axis
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
                
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=10)
        plt.tight_layout()

        # Adding a title to the figure
        if title is not None:
            fig.suptitle(title)

        if now :
            plt.show()
    else :
        fig = None
        ax = None

    return df_mean, df_std, fig, ax

# =============================================================================

def plot_features(df, ax=None, norm=True, mode="24h", **kwargs):
    """
    Plot the variation of features values (ie. indices) in the DataFrame obtained 
    with ``maad.features``.
            
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
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.
            
        - yticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.

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
    see plot_extract_alpha_indices.py advanced example for a complete example
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> np.random.seed(2021)
    >>> M = np.random.rand(24, 2)
    >>> df = pd.DataFrame(M)
    >>> indices = ['A','B']
    >>> df.columns = indices
    >>> df.index =pd.date_range(start=pd.Timestamp('00:00:00'), end=pd.Timestamp('23:00:00'), freq='1H')
    >>> maad.util.plot_features(df) 
        
    """
    if isinstance(df, pd.DataFrame) == False:
        raise TypeError("df must be a Pandas Dataframe")
    elif isinstance(df.index, pd.DatetimeIndex) == False:
        raise TypeError("df must have an index of type DateTimeIndex")

    if mode == "24h":
        # Mean values by hour
        df = df.groupby(df.index.hour).mean()

    if norm:
        df = linear_scale(df)

    # plot
    import itertools
    from matplotlib.lines import Line2D

    list_markers = tuple(list(Line2D.markers.keys())[0:-4])
    markers = itertools.cycle(list_markers)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = itertools.cycle(prop_cycle.by_key()["color"])

    figsize = kwargs.pop("figsize", (5, 5))
    kwargs.pop("label", None)
    now = kwargs.pop("now", True)

    # if no ax, create a figure and a subplot associated a figure otherwise
    # find the figure that belongs to ax
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
        fig.set_size_inches(figsize)
    else:
        fig = ax.get_figure()

    for indice in list(df):
        ax.plot(
            df.index,
            df[indice],
            marker=next(markers),
            color=next(colors),
            linestyle="-",
            label=indice,
            **kwargs
        )

    if mode == "24h":
        ax.set_xlabel("Day time (Hour)")
    ax.grid()
    ax.legend()
    fig.tight_layout()

    # Display the figure now
    if now:
        plt.show()

    return fig, ax


# =============================================================================
def plot_correlation_map(df, R_threshold=0.75, method="spearman", **kwargs):
    """
    Plot the correlation map between indices in the DataFrame obtained
    with ``maad``.
        
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
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.
            
        - yticks : tuple of ndarrays, optional, default : none
            * ticks : array_like => A list of positions at which ticks should be placed. You can pass an empty list to disable yticks.
            * labels : array_like, optional =>  A list of explicit labels to place at the given locs.
        
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
    see plot_extract_alpha_indices.py advanced example for a complete example
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> np.random.seed(2021)
    >>> M = np.random.rand(10, 10)
    >>> df = pd.DataFrame(M)
    >>> indices = ['A','B','C','D','E','F','G','H','I','J']
    >>> df.columns = indices
    >>> maad.util.plot_correlation_map(df, R_threshold=0)
    """
    # Correlation matrix with only the columns that contain numeric values
    numeric_columns = df.select_dtypes(include='number')
    corr_matrix = numeric_columns.corr()

    # pop kwargs
    figsize = kwargs.pop("figsize", (10, 8))
    cmap = kwargs.pop("cmap", "RdBu_r")
    label_colorbar = kwargs.pop("label_colorbar", "R")

    fig = plt.figure()
    fig.set_size_inches(figsize)
    ax = fig.add_subplot(111)
    caxes = ax.matshow(corr_matrix[abs(corr_matrix) ** 2 > R_threshold ** 2], cmap=cmap)
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


# =============================================================================
def false_Color_Spectro(
    df,
    indices=None,
    plim=(1, 99),
    reverseLUT=False,
    permut=False,
    unit="minutes",
    verbose=False,
    display=False,
    savefig=None,
    **kwargs
):
    """
    Create False Color Spectrogram from indices obtained by MAAD.
    Only indices than can be computed bin per bin (ie frequency per frequency)
    are used to create False Color Spectro. They are called xxx_per_bin.
        
    Parameters
    ----------
    df : Panda DataFrame
        DataFrame with indices per frequency bin.
        
    indices : list, default is None
        List of indices. 
        If permut is False (see permut), if indices is None : 1st indices is red (R), 2nd indice is green (G) 
        and the 3rd indices is blue (B).
        if indices is a list of 3 indices or more, only the 3 first indices (triplet) are used to create the false color spectro.
        if permut is True, if indices is None : All indices in df are used
        if indices is a list of 3 indices or more, all the indices in the 
        list are used to create the false color spectro with all the possibile permutations.
    
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
    
    kwargs, optional
       - dpi : scalar, optional, default 96
       - format : str, optional, default .png
    
    Returns
    -------
    false_color_image : ndarray of scalars
        Matrix with ndim = 4 if multiple false color spectro or ndim = 3, if
        single false color spectro with 3 colors : R, G, B
            
    triplet : list
        List of triplet of indices corresponding to each false color spectro
        
    Examples
    --------
    see plot_extract_alpha_indices.py advanced example for a complete example 
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> A_per_bin = [[0.5,0.4,0.3,0.2,0.1],
                     [0.3,0.3,0.3,0.3,0.3],
                     [0.5,0.5,0.3,0.0,0.0],
                     [0.2,0.2,0.3,0.0,0.0],
                     [0.0,0.0,0.3,0.0,0.0]]
    >>> B_per_bin = [[0.5,0.5,0.5,0.5,0.5],
                     [0.5,0.5,0.5,0.5,0.5],
                     [0.5,0.5,0.2,0.0,0.0],
                     [0.5,0.2,0.1,0.0,0.0],
                     [0.5,0.2,0.0,0.0,0.0]]
    >>> C_per_bin = [[0.0,0.7,0.0,0.5,0.5],
                     [0.7,0.2,0.0,0.5,0.5],
                     [0.5,0.5,0.7,0.2,0.2],
                     [0.0,0.7,0.0,0.2,0.2],
                     [0.7,0.2,0.0,0.2,0.2]]   
    >>> df = pd.DataFrame([A_per_bin,B_per_bin,C_per_bin])
    >>> df = df.T
    >>> df.columns = ['A_per_bin','B_per_bin','C_per_bin']
    >>> df.index = pd.date_range('20200101', periods=len(df))
    >>> maad.util.false_Color_Spectro (df, display=True ,unit='days', figsize=[3,3])
    
    """
    # sort dataframe by date
    df = df.sort_index(axis=0)
    # set index as DatetimeIndex
    df = df.set_index(pd.DatetimeIndex(df.index))
    # remove column file
    if "file" in df.columns:
        df = df.drop(columns="file")
    # test if frequencies is in columns
    if "frequencies" in df.columns:
        # get frequencies and remove the column
        # test if type of df['frequencies'] is number or str
        if isinstance(df.frequencies.iloc[0], str):
            fn = df["frequencies"].apply(literal_eval).iloc[0]
        else:
            fn = df["frequencies"].iloc[0]
        # drop frequencies
        df = df.drop(columns="frequencies")
    else:
        fn = np.arange(0, len(df))

    # Set the list of indices if indices is None
    if indices is None:
        indices = list(df)

    # for each indice, check if values type is string and convert into scalars
    # Values are vectors of strings when data are loading from csv
    for indice in indices:
        # test if type of df[indice] is string
        if isinstance(df[indice].iloc[0], str):
            # convert string into scalars
            df[indice] = df[indice].apply(literal_eval)
            
    # create a dataframe with normalized values for each indices
    df_z = pd.DataFrame()

    for indice in indices:
        z = []
        if verbose:
            print(indice)
        for v in df[indice]:
            z.append(v)
        z = np.asarray(z).T
        # Select the min and max value for each indice
        z_min = np.percentile(z, plim[0])
        z_max = np.percentile(z, plim[1])
        # clip the value to the min and max found
        z = np.clip(z, z_min, z_max)
        # linear conversion
        if reverseLUT == True:
            # between 1 to 0
            z = linear_scale(z, 1, 0)
        else:
            # between 0 to 1
            z = linear_scale(z, 0, 1)

        df_z[indice] = [z * 255]

    # find all permutation of 3 indices among all indices
    if permut == True:
        import itertools

        per = itertools.permutations(list(df_z), 3)
        triplet = []
        for val in per:
            triplet.append([*val])
    else:
        triplet = []
        triplet.append(indices[0])
        triplet.append(indices[1])
        triplet.append(indices[2])
        triplet = [triplet]

    #####################

    # get the number of pixels along frequency (Nf) and time (Nt)
    Nf, Nt = df_z[triplet[0][0]].values.tolist()[0].shape

    # test if figsize is in kwargs
    figsize = kwargs.pop("figsize", None)
    # if figsize is not in kwargs
    if figsize is None:
        figsize = (6 * Nt / 250, 10 * Nf / 512)

    fig_kwargs = {"figsize": figsize, "tight_layout": "tight_layout"}

    # number of days in the period
    deltaT = df.index.max() - df.index.min()

    # unit
    if unit == "minutes":
        normT = 60e9
        xlabel = "Minutes"
    elif unit == "hours":
        normT = 60e9 * 60
        xlabel = "Hours"
    elif unit == "days":
        normT = 60e9 * 60 * 24
        xlabel = "Days"
    elif unit == "weeks":
        normT = 60e9 * 60 * 24 * 7
        xlabel = "Weeks"
    else:
        normT = 60e9
        xlabel = "Minutes"

    false_color_image = []
    for tt in np.arange(len(triplet)):

        # create the false color image (R,G,B)
        z0 = df_z[triplet[tt][0]].values.tolist()[0]
        z1 = df_z[triplet[tt][1]].values.tolist()[0]
        z2 = df_z[triplet[tt][2]].values.tolist()[0]
        false_color_image.append((np.dstack((z0, z1, z2))).astype(np.uint8))

        # Display the False Color Spectro
        if display:
            plt.rcParams.update({"font.size": 10})
            plt.rcParams.update({"font.family": "serif"})
            fig = plt.figure(facecolor="white", **fig_kwargs)
            plt.imshow(
                false_color_image[tt],
                aspect="auto",
                origin="lower",
                extent=(fn[0], deltaT.value / normT, 0, fn[-1]),
                **kwargs
            )
            plt.xlabel(xlabel)
            plt.ylabel("Frequency (Hz)")
            plt.title(
                "False Color Spectro "
                + "\n"
                + " [R:"
                + triplet[tt][0][:-8]
                + "; "
                + "G:"
                + triplet[tt][1][:-8]
                + "; "
                + "B:"
                + triplet[tt][2][:-8]
                + "]",
                size=12,
            )
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()

            # SAVE FIGURE
            if savefig is not None:
                dpi = kwargs.pop("dpi", 96)
                bbox_inches = "tight"
                format = kwargs.pop("format", "png")
                filename = (
                    "_fcs_"
                    + triplet[tt][0][:-8]
                    + "_"
                    + triplet[tt][1][:-8]
                    + "_"
                    + triplet[tt][2][:-8]
                )
                full_filename = savefig + filename + "." + format
                if verbose:
                    print("\n" "save figure : %s" % full_filename)
                # save fig
                fig.savefig(
                    fname=full_filename,
                    dpi=dpi,
                    bbox_inches=bbox_inches,
                    format=format,
                    **kwargs
                )
                # close fig
                plt.close(fig)

    # convert into ndarray
    false_color_image = np.asarray(false_color_image)

    # test if there is only one False Color Spectro, then, remove the 1st dim
    if false_color_image.shape[0] == 1:
        false_color_image = false_color_image[0]
        triplet = triplet[0]

    return false_color_image, triplet
