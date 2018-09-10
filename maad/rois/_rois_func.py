#!/usr/bin/env python
""" functions for processing ROIS """
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
from math import ceil
from scipy import signal
from scipy.ndimage import morphology
from scipy.stats import iqr
from skimage import measure, filters
from skimage.io import imread

import matplotlib.patches as mpatches
from ..util import plot1D, plot2D, linear_scale,read_audacity_annot,rand_cmap

"""
====== TO DO
"""


def select_bandwidth():
    return
"""
====== 
"""

"""****************************************************************************
************* Load an image and convert it in gray level if needed  ***********
****************************************************************************"""
def load(filename, fs, d, flipud = True, display=False, **kwargs):
    """
    Load an image from a file or an URL
    
    Parameters
    ----------  
    filename : string
        Image file name, e.g. ``test.jpg`` or URL.
    Returns
    -------
    im : ndarray
        The different color bands/channels are stored in the
        third dimension, such that a gray-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.
    """
    # Load image
    im  = imread(filename, as_gray=True)
    
    # if 3D, convert into 2D
    if len(im.shape) == 3:
        im = im[:,:,0]
        
    # Rescale the image between 0 to 1
    im = linear_scale(im, minval= 0.0, maxval=1.0)
            
    # Get the resolution
    df = fs/(im.shape[0]-1)
    dt = d/(im.shape[1]-1)
    
    # Extent
    ext = [0, d, 0, fs/2]
    
    # flip the image vertically
    if flipud: im = np.flip(im, 0)
    
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','loaded spectrogram')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        
        _, fig = plot2D (im, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax,
                         cmap=cmap, **kwargs)
    
    return im, ext, dt, df

"""****************************************************************************
*************               noise_subtraction                       ***********
****************************************************************************"""
def remove_background(im, ext, gauss_win=50, gauss_std = 25, beta1=1, beta2=1, 
                      llambda=1, display = False, savefig=None, **kwargs):
    """
    Remove the background noise using spectral subtraction

    Based on the spectrum of the A posteriori noise profile.  
    It computes an atenuation map in the time-frequency domain. 
    See [1] or [2] for more detail about the algorithm.

    References:

    [1] Steven F. Boll, "Suppression of Acoustic Noise in Speech Using Spectral
    Subtraction", IEEE Transactions on Signal Processing, 27(2),pp 113-120,
    1979

    [2] Y. Ephraim and D. Malah, Speech enhancement using a minimum mean square 
    error short-time spectral amplitude estimator, IEEE. Transactions in
    Acoust., Speech, Signal Process., vol. 32, no. 6, pp. 11091121, Dec. 1984.

    Parameters
    ----------
    im : 2d ndarray of scalar
        Spectrogram 

    gauss_win=50 : int, optional, default: 50
        Number of points in the gaussian window 
        
    gauss_std = 25
        The standard deviation, sigma used to create the gaussian window  
        
    beta1 : scaler, optional, default: 1      
        beta1 has to be >0
        Should be close to 1

    beta2: scaler, optional, default: 1       
        beta2 has to be >0 
        better to not change
    
    llambda : int, optional, default: 1       
        over-subtraction factor to compensate variation of noise amplitude.
        Should be close to 1
                      
    Returns
    -------
    Sxx_out : 2d ndarray of scalar
        Spectrogram after denoising           
    """  
    
    Nf, Nw = im.shape
          
    # noise spectrum extraction
    absS=np.abs(im)**2;
    mean_profile=np.mean(absS,1) # average spectrum (assumed to be ergodic)
    if display == True:
        if ext is not None :
            #fn = np.arange(ext[2], ext[3]+(ext[3]-ext[2])/(Nf-1), (ext[3]-ext[2])/(Nf-1))
            fn = np.arange(0, Nf)*(ext[3]-ext[2])/(Nf-1) + ext[2] 
            xlabel = 'frequency [Hz]'
        else:
            fn = np.arange(Nf)
            xlabel = 'pseudofrequency [points]'
        ax1,_ = plot1D(fn, mean_profile, figtitle = 'Noise profile',
                               xlabel = xlabel, ylabel = 'Amplitude [AU]',
                               linecolor = 'r')  
    
    # White Top Hat (to remove non uniform background) = i - opening(i)
    selem = signal.gaussian(gauss_win, gauss_std)
    noise_profile = morphology.grey_opening(mean_profile, structure=selem)
    if display == True: 
        if ext is not None :
            #fn = np.arange(ext[2], ext[3]+(ext[3]-ext[2])/(Nf-1), (ext[3]-ext[2])/(Nf-1))
            fn = np.arange(0, Nf)*(ext[3]-ext[2])/(Nf-1) + ext[2] 
            xlabel = 'frequency [Hz]'
        else:
            fn = np.arange(Nf)
            xlabel = 'pseudofrequency [points]'
        plot1D(fn, noise_profile, ax =ax1, figtitle = 'Noise profile',
                               xlabel = xlabel, ylabel = 'Amplitude [AU]',
                               linecolor = 'k')  

    # Remove the artefact at the end of the spectrum (2 highest frequencies)
    noise_profile[-2:] = mean_profile [-2:]
    noise_profile[:2] = mean_profile [:2]
    
    # Create a matrix with the noise profile
    noise_spectro=np.kron(np.ones((Nw,1)),noise_profile)
    noise_spectro=noise_spectro.transpose()
        
    # snr estimate a posteriori
    SNR_est=(absS/noise_spectro) -1  
    SNR_est=SNR_est*(SNR_est>0) # keep only positive values
    
    # compute attenuation map
    # if llambda, beta1 and beta 2 are equal to 1, it is (1 - noise_spectro)
    an_lk=(1-llambda*((1./(SNR_est+1))**beta1))**beta2
    an_lk=an_lk*(an_lk>0) # keep only positive values
    
    # Apply the attenuation map to the STFT coefficients
    im_out=an_lk*im
    
    # if nan in the image, convert nan into 0
    np.nan_to_num(im_out,0)
                
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Spectrogram without stationnary noise')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        
        _, fig = plot2D (im_out, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax,
                         cmap=cmap, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png') 
            filename=kwargs.pop('filename','_spectro_after_noise_subtraction')                
            filename = savefig+filename+'.'+format
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs) 

    return im_out  

"""****************************************************************************
*************                      blurr                            ***********
****************************************************************************"""
def blurr(im, ext, std=1, display = False, savefig=None, **kwargs):
    """
    Aim:blurr the image before binarisation
    INPUT
        im: input image 
        ext:
        std: std of the gaussian filter
    OUTPUT
        im_out: blurred image 
    """
    # use scikit image (faster than scipy)
    im_out = filters.gaussian(im,std)
    
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Blurred spectrogram')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        
        _, fig = plot2D (im_out, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax,
                         cmap=cmap, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png')
            filename=kwargs.pop('filename','_spectro_blurr')             
            filename = savefig+filename+'.'+format
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs)   
    
    return im_out

"""****************************************************************************
*************                double_threshold                       ***********
****************************************************************************"""
def _double_threshold_rel (im, ext, bin_std=5, bin_per=0.1, display=False, savefig=None,
                     **kwargs):
    """
    from MATLAB
    Threshold estimation (from Oliveira et al, 2015)
    % bin_std : to set the first threshold. It ajusts a sort of std. Value should be around 0.5 to 3 
    % bin_per: amount to set second threshold lower. From 0 to 1. ex: 0.1 = 10 %
    Adapted by S. Haupert Dec 12, 2017
    """
    # Compute the qth percentile of the data along the specified axis
    val1 = np.percentile(im[np.where(im>0)],75)   # value corresponding to the limit between the 75% lowest value and 25% largest value
    
    # The interquartile range (IQR) is the difference between the 75th and 
    # 25th percentile of the data. It is a measure of the dispersion similar 
    # to standard deviation or variance, but is much more robust against outliers
    val2 = iqr(im[np.where(im>0)])*bin_std
    
    # Threshold : qth  percentile + sort of std
    h_th = val1 + val2
    print ('high threshold value %.2f' % h_th)
    
    # Low threshold limit
    l_th = (h_th-h_th*bin_per)
    print ('low threshold value %.2f' % l_th)
    
    # binarisation 
    im_t1 = im > h_th    # mask1
    im_t2 = im > l_th    # mask2
    im_t3 = im * im_t1   # selected parts of the image
    
    #find index of regions which meet the criteria
    conncomp_t2 = measure.label(im_t2)    #Find connected components in binary image
    rprops = measure.regionprops(conncomp_t2,im_t3) 
    
    rprops_mean_intensity = [region['mean_intensity'] for region in rprops]  
    rprops_mean_intensity = np.asarray(rprops_mean_intensity)
    
    rprops_label = [region['label'] for region in rprops] 
    rprops_label = np.asarray(rprops_label)
        
    [ind]=np.where(rprops_mean_intensity>0)
    
    im_out = np.isin(conncomp_t2, rprops_label[ind])    # test if the indice is in the matrix of indices
    im_out =im_out*1    #  boolean to 0,1 conversion
                
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','binary image => MASK')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        
        _, fig = plot2D (im_out, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax,
                         cmap=cmap, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png') 
            filename=kwargs.pop('filename','_spectro_binary')                
            filename = savefig+filename+'.'+format
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs) 

    return im_out

"""****************************************************************************
*************                double_threshold                       ***********
****************************************************************************"""
def _double_threshold_abs(im, ext, bin_h=0.7, bin_l=0.2, display=False, savefig=None,
                     **kwargs):
    """
    % bin_h : 
    % bin_l: 

    """
    
    # binarisation 
    im_t1 = im > bin_h  # mask1
    im_t2 = im > bin_l  # mask2
    im_t3 = im * im_t1  # selected parts of the image
    
    #find index of regions which meet the criteria
    conncomp_t2 = measure.label(im_t2)    #Find connected components in binary image
    rprops = measure.regionprops(conncomp_t2,im_t3) 
    
    rprops_mean_intensity = [region['mean_intensity'] for region in rprops]  
    rprops_mean_intensity = np.asarray(rprops_mean_intensity)
    
    rprops_label = [region['label'] for region in rprops] 
    rprops_label = np.asarray(rprops_label)
        
    [ind]=np.where(rprops_mean_intensity>0)
    
    im_out = np.isin(conncomp_t2, rprops_label[ind])    # test if the indice is in the maxtrix of indices
    im_out =im_out*1    #  boolean to 0,1 conversion
                
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','binary image => MASK')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        
        _, fig = plot2D (im_out, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax,
                         cmap=cmap, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png') 
            filename=kwargs.pop('filename','_spectro_binary')                
            filename = savefig+filename+'.'+format
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs) 

    return im_out

"""****************************************************************************
*************                   create_mask                         ***********
****************************************************************************"""
def create_mask(im, ext, display = False, savefig = None, **kwargs):
    
    mode=kwargs.pop('mode', 'relative')
    
    if mode == 'relative':
        bin_std=kwargs.pop('bin_std', 3) 
        bin_per=kwargs.pop('bin_per', 0.9) 
        print ('create binary mask with mode relative (bin_std=%.2f, bin_per=%.2f)' % (bin_std, bin_per))
        im_bin = _double_threshold_rel(im, ext, bin_std, bin_per, display, savefig, **kwargs)
        
    elif mode == 'absolue':
        bin_h=kwargs.pop('bin_h', 0.7) 
        bin_l=kwargs.pop('bin_l', 0.3) 
        im_bin = _double_threshold_abs(im, ext, bin_h, bin_l, display, savefig, **kwargs)   
    
    return im_bin 

"""****************************************************************************
*************                   select_rois                         ***********
****************************************************************************"""
def select_rois(im_bin, ext, min_roi=None ,max_roi=None, display=False, 
                savefig = None, **kwargs):
    """
    Aim: Select rois candidates based on area of rois. min and max boundaries
    INPUT
        im_bin: a binary image mask
        ext: scale (xmin, xmax, ymin, ymax)
        min_roi and max_roi: boundaries to select rois. rois with values under
        min_roi and over max_roi will be removed.
    OUTPUT
        im_label: image with selected rois (label)
    """

    labels = measure.label(im_bin)    #Find connected components in binary image
    rprops = measure.regionprops(labels)
    
    rois_bbox = []
    rois_label = []
    
    for roi in rprops:
        
        # select the rois  depending on their size
        if (roi.area >= min_roi) & (roi.area <= max_roi):
            # get the label
            rois_label.append(roi.label)
            # get rectangle coordonates           
            rois_bbox.append (roi.bbox)    
                
    im_rois = np.isin(labels, rois_label)    # test if the indice is in the matrix of indices
    im_rois = im_rois* labels
    
    # create a list with labelID and labelName (None in this case)
    rois_label = list(zip(rois_label,['unknown']*len(rois_label)))
    
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Selected ROIs') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        
        randcmap = rand_cmap(len(rois_label))
        cmap   =kwargs.pop('cmap',randcmap) 
        
        _, fig = plot2D (im_rois, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,
                         cmap=cmap, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png') 
            filename=kwargs.pop('filename','_spectro_selectrois')                
            filename = savefig+filename+'.'+format
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs) 
 
    return im_rois, rois_bbox, rois_label  

"""****************************************************************************
*************                 select_rois_manually                  ***********
****************************************************************************"""
def select_rois_manually(im_bin, ext, filename, software='audacity', mask=True, display=False, 
                savefig = None, **kwargs):
    """
    Aim: Select rois candidates based on area of rois. min and max boundaries
    INPUT
        im_bin: a binary image mask
        ext: scale (xmin, xmax, ymin, ymax)
        min_roi and max_roi: boundaries to select rois. rois with values under
        min_roi and over max_roi will be removed.
    OUTPUT
        im_label: image with selected rois (label)
    """
    Nf, Nt = im_bin.shape
    df = (ext[3]-ext[2])/(Nf-1)
    dt = (ext[1]-ext[0])/(Nt-1)
    t0 = ext[0]
    f0 = ext[2] 
    
    # get the offset in x and y depending on the starting time and freq
    offset_x = np.round((t0)/dt).astype(int)
    offset_y = np.round((f0)/df).astype(int)  

    im_rois = np.zeros(im_bin.shape).astype(int) 
    rois_bbox = []
    rois_label = []
    if software=='audacity':
        tab_out = read_audacity_annot(filename)
        
        ymin = (tab_out['fmin']/df+offset_y).astype(int)
        xmin = (tab_out['tmin']/dt+offset_x).astype(int)
        ymax = (tab_out['fmax']/df+offset_y).astype(int)
        xmax = (tab_out['tmax']/dt+offset_x).astype(int)
        zipped = zip(ymin,xmin,ymax,xmax)
        
        # bbox
        bbbox= list(zipped)
        # add current bbox to the list of bbox
        rois_bbox.extend(bbbox)        

    # Construction of the ROIS image with the bbox and labels
    index = 0
    labelID = []
    labelName = []
    for ymin, xmin, ymax, xmax in bbbox:
        index= index +1
        im_rois[ymin:ymax,xmin:xmax] = index
        labelName.append(tab_out['label'][index-1])
        labelID.append(index)
        
    rois_label = list(zip(labelID,labelName)) 

    rois_label_man = []
    rois_bbox_man= []
    if mask:
        im_rois = (im_bin * im_rois).astype(int)
        # Select only rois_label that are present into im_rois
        for index, (ID, LABEL) in enumerate (rois_label): 
            if ID in im_rois:
                 rois_label_man.append(rois_label[index])
                 rois_bbox_man.append(rois_bbox[index])
    else:
        rois_label_man = rois_label
        rois_bbox_man = rois_bbox
                
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Selected ROIs')
        figsize=kwargs.pop('figsize',(4, 13)) 
        
        randcmap = rand_cmap(len(np.unique(rois_label_man)))
        cmap   =kwargs.pop('cmap',randcmap) 
        
        _, fig = plot2D (im_rois, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,
                         cmap=cmap, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png') 
            filename=kwargs.pop('filename','_spectro_selectrois')                
            filename = savefig+filename+'.'+format
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs) 
      
    return im_rois, rois_bbox_man, rois_label_man 

"""****************************************************************************
*************                   display_rois                        ***********
****************************************************************************"""
def overlay_rois (im_ref, ext, rois_bbox, rois_label=None, savefig=None, **kwargs):
     
    
    ylabel =kwargs.pop('ylabel','Frequency [Hz]')
    xlabel =kwargs.pop('xlabel','Time [sec]') 
    title  =kwargs.pop('title','ROIs Overlay')
    cmap   =kwargs.pop('cmap','gray') 
    figsize=kwargs.pop('figsize',(4, 13)) 
    vmin=kwargs.pop('vmin',0) 
    vmax=kwargs.pop('vmax',1) 
        
    ax, fig = plot2D (im_ref, extent=ext, figsize=figsize,title=title, 
                     ylabel = ylabel, xlabel = xlabel,vmin=vmin,vmax=vmax, 
                     cmap=cmap, **kwargs)

    # Convert pixels into time and frequency values
    y_len, x_len = im_ref.shape
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim() 
    x_scaling = (xmax-xmin) / x_len
    y_scaling = (ymax-ymin) / y_len
    

    if rois_label is None:
        for y0, x0, y1, x1 in rois_bbox:
            rect = mpatches.Rectangle((x0*x_scaling+xmin, y0*y_scaling+ymin), 
                                      (x1-x0)*x_scaling, 
                                      (y1-y0)*y_scaling,
                                      fill=False, edgecolor='yellow', linewidth=1)  
            # draw the rectangle
            ax.add_patch(rect)
    else :
        # Colormap
        labelID, labelName = zip(*rois_label)
        labelNames = np.unique(np.array(labelName))
        color = rand_cmap(len(labelNames)+1,first_color_black=False) 
        for bbox, label in zip(rois_bbox, rois_label):
            y0 = bbox[0]
            x0 = bbox[1]
            y1 = bbox[2]
            x1 = bbox[3]
            for index, name in enumerate(labelNames):
                if label[1] in name: ii = index
            rect = mpatches.Rectangle((x0*x_scaling+xmin, y0*y_scaling+ymin), 
                                      (x1-x0)*x_scaling, 
                                      (y1-y0)*y_scaling,
                                      fill=False, edgecolor=color(ii), linewidth=1)  
            # draw the rectangle
            ax.add_patch(rect)

    fig.canvas.draw()
    
    # SAVE FIGURE
    if savefig is not None : 
        dpi   =kwargs.pop('dpi',96)
        format=kwargs.pop('format','png') 
        filename=kwargs.pop('filename','_spectro_overlayrois')                
        filename = savefig+filename+'.'+format
        fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                    **kwargs) 
    
    return

"""****************************************************************************
*************                   find_rois                           ***********
****************************************************************************"""
def find_rois_wrapper(im, ext, display=False, savefig=None, **kwargs):

    df = (ext[3]-ext[2])/(im.shape[0]-1)
    dt = (ext[1]-ext[0])/(im.shape[1]-1)
    
    gauss_win=kwargs.pop('gauss_win',round(500/df))
    gauss_std=kwargs.pop('gauss_std',gauss_win*2)
    beta1=kwargs.pop('beta1', 0.9)
    beta2=kwargs.pop('beta2', 1)
    llambda=kwargs.pop('llambda', 1.5)  
       
    mode=kwargs.pop('mode', 'relative') 
        
    min_roi=kwargs.pop('min_roi', None)
    if min_roi is None:
        min_f = ceil(100/df) # 100Hz 
        min_t = ceil(0.1/dt) # 100ms 
        min_roi=np.min(min_f*min_t) 
        
    max_roi=kwargs.pop('max_roi', None) 
    if max_roi is None:    
        max_f = np.asarray([round(1000/df), im.shape[0]])
        max_t = np.asarray([im.shape[1], round(1/dt)])     
        max_roi =  np.max(max_f*max_t)

    im_denoized = remove_background(im, ext, gauss_win=gauss_win, 
                                    gauss_std=gauss_std, beta1=beta1, 
                                    beta2=beta2, llambda=llambda, 
                                    display= display, savefig=savefig, **kwargs)
    
    im_bin = create_mask(im_denoized, ext, mode=mode, display=display, 
                         savefig=savefig, **kwargs)
        
    im_rois, rois_bbox  = select_rois(im_bin, ext, min_roi=min_roi, 
                                       max_roi=max_roi, display=display, 
                                       savefig=savefig, **kwargs)
    
    if display: overlay_rois(im, ext, rois_bbox, savefig=savefig, **kwargs)

    return im_rois, rois_bbox
