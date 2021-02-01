#!/usr/bin/env python 
"""  
Segmentation methods to find regions of interest in the time and frequency domain.
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
from scipy.stats import iqr 
from skimage import measure
import pandas as pd 
import sys 
_MIN_ = sys.float_info.min 
 
# Import internal modules 
from maad.util import (plot2d, rand_cmap)

#%%
#**************************************************************************** 
# private functions                
#**************************************************************************** 
def _double_threshold_rel (im, bin_std=6, bin_per=0.5, 
                           verbose=False, display=False, savefig=None, **kwargs): 
    """ 
    Binarize an image based on a double relative threshold.  
    The values used for the thresholding depends on the values found in the  
    image. => relative threshold 
     
    Parameters 
    ---------- 
    im : 2d ndarray of scalars 
        Spectrogram (or image) 
 
    bin_std : scalar, optional, default is 6 
        Set the first threshold. This threshold is not an absolute value but 
        depends on values that are similar to 75th percentile (pseudo_mean) and 
        a sort of std value of the image.  
        threshold1 = "pseudo_mean" + "std" * bin_std    
        Value higher than threshold1 are set to 1, they are the seeds for  
        the second step. The others are set to 0.  
         
    bin_per: scalar, optional, defautl is 0.5 
        Set how much the second threshold is lower than the first 
        threshold value. From 0 to 1. ex: 0.1 = 10 %. 
        threshold2 = threshold1 (1-bin_per)   
        Value higher than threshold2 and connected (directly or not) to the  
        seeds are set to 1, the other remains 0 
 
    verbose : boolean, optional, default is False
        print messages
    
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
             
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
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
    im_out: binary image  
     
    References 
    ---------- 
    .. [1] from MATLAB: Threshold estimation (from Oliveira et al, 2015) 
       Adapted by S. Haupert Dec 12, 2017 
    """ 
    
    # test if im is full of zeros
    if not im.any() :
        im_out = np.zeros(im.shape)   
    else:
    
        # Compute the qth percentile of the data along the specified axis 
        val1 = np.percentile(im[np.where(im>0)],75)   # value corresponding to the limit between the 75% lowest value and 25% largest value 
         
        # The interquartile range (IQR) is the difference between the 75th and  
        # 25th percentile of the data. It is a measure of the dispersion similar  
        # to standard deviation or variance, but is much more robust against outliers 
        val2 = iqr(im[np.where(im>0)])*bin_std 
         
        # Threshold : qth  percentile + sort of std 
        h_th = val1 + val2 
        # Low threshold limit 
        l_th = (h_th-h_th*bin_per) 
        
        if verbose :
            print(72 * '_') 
            print('Double thresholding with values relative to the image...') 
            print ('**********************************************************') 
            print ('  high threshold value %.2f | low threshold value %.2f' % (h_th, l_th)) 
            print ('**********************************************************') 
         
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
            vmin=kwargs.pop('vmin',0)  
            vmax=kwargs.pop('vmax',1)  
            extent=kwargs.pop('extent',None)
                
            if extent is not None : 
                xlabel = 'frequency [Hz]' 
                figsize=kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
            else: 
                xlabel = 'pseudofrequency [points]'
                figsize=kwargs.pop('figsize',(4, 13)) 
             
            _, fig = plot2d (im_out, extent=extent, figsize=figsize,title=title,  
                             ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                             cmap=cmap, **kwargs) 
            # SAVE FIGURE 
            if savefig is not None :  
                dpi   =kwargs.pop('dpi',96) 
                format=kwargs.pop('format','png') 
                filename=kwargs.pop('filename','_spectro_binary')              
                filename = savefig+filename+'.'+format 
                if verbose :
                    print('\n''save figure : %s' %filename) 
                fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format, 
                            **kwargs)    
 
    return im_out 
 
#%% 
def _double_threshold_abs(im, bin_h=0.7, bin_l=0.2, 
                          verbose=False,display=False, savefig=None, **kwargs): 
    """ 
    Binarize an image based on a double relative threshold.  
    The values used for the thresholding are independent of the values in the 
     
    image => absolute threshold 
     
    Parameters 
    ---------- 
    im : 2d ndarray of scalars 
        Spectrogram (or image) 

    bin_h : scalar, optional, default is 0.7 
        Set the first threshold. Value higher than this value are set to 1,  
        the others are set to 0. They are the seeds for the second step 
         
    bin_l: scalar, optional, defautl is 0.2 
        Set the second threshold. Value higher than this value and connected 
        to the seeds or to other pixels connected to the seeds are set to 1,  
        the other remains 0 
        
    verbose : boolean, optional, default is False
        print messages
 
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
           
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
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
    im_out: binary image  
     
    References 
    ---------- 
    .. [1] J. Canny. A computational approach to edge detection. IEEE 
    Transactions on Pattern Analysis and Machine Intelligence. 1986; vol. 8, 
    pp.679-698. DOI:10.1109/TPAMI.1986.4767851 
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
    
    if verbose :
        print(72 * '_') 
        print('Double thresholding with  absolute values...') 
        print ('**********************************************************') 
        print ('  Number of rois %.2f | Rois cover %.2f%' % (len(rprops_label), 
                                                             sum(im_out)/(im_out.shape[1]*im_out.shape[0])*100)) 
        print ('**********************************************************') 
                 
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')  
        title  =kwargs.pop('title','binary image => MASK') 
        cmap   =kwargs.pop('cmap','gray')  
        vmin=kwargs.pop('vmin',0)  
        vmax=kwargs.pop('vmax',1)  
        extent=kwargs.pop('extent',None)
            
        if extent is not None : 
            xlabel = 'frequency [Hz]' 
            figsize=kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
        else: 
            xlabel = 'pseudofrequency [points]'
            figsize=kwargs.pop('figsize',(4, 13)) 
         
        _, fig = plot2d (im_out, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
        # SAVE FIGURE 
        if savefig is not None :  
            dpi   =kwargs.pop('dpi',96) 
            format=kwargs.pop('format','png') 
            filename=kwargs.pop('filename','_spectro_binary')              
            filename = savefig+filename+'.'+format 
            if verbose :
                print('\n''save figure : %s' %filename) 
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format, 
                        **kwargs)    
 
    return im_out 
 
#%%
# =============================================================================
# public functions
# =============================================================================    
def create_mask(im, mode_bin = 'relative', 
                verbose= False, display = False, savefig = None, **kwargs): 
    """ 
    Binarize an image based on a double threshold.  
     
    Parameters 
    ---------- 
    im : 2d ndarray of scalars 
        Spectrogram (or image) 
 
    mode_bin : string in {'relative', 'absolute'}, optional, default is 'relative' 
        if 'absolute' [1]_ , a double threshold with absolute value is performed 
        with two parameters (see \*\*kwargs section)
        if 'relative' [2]_, a relative double threshold is performed with two 
        parameters (see \*\*kwargs section)

    verbose : boolean, optional, default is False
        print messages
         
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by the maad functions as well 
        as the plt.plot and savefig functions. 
        All the input arguments required or optional in the signature of the 
        functions above can be passed as kwargs : 
        
        if 'absolute' [1]_ 
        - bin_h : scalar, optional, default is 0.7 
        Set the first threshold. Value higher than this value are set to 1,  
        the others are set to 0. They are the seeds for the second step 
        - bin_l: scalar, optional, defautl is 0.2 
        Set the second threshold. Value higher than this value and connected 
        to the seeds or to other pixels connected to the seeds (6-connectivity)
        are set to 1, the other remains 0  
        
        if 'relative' [2]_ :
        - bin_std :  scalar, optional, default is 6 
        bin_std is needed to compute the threshold1. 
        This threshold is not an absolute value but depends on values that are 
        similar to 75th percentile (pseudo_mean) and a sort of std value of 
        the image.  
        threshold1 = "pseudo_mean" + "std" * bin_std    
        Value higher than threshold1 are set to 1, they are the seeds for  
        the second step. The others are set to 0.  
        - bin_per: scalar, optional, defautl is 0.5 
        Set how much the second threshold is lower than the first 
        threshold value. From 0 to 1. ex: 0.1 = 10 %. 
        threshold2 = threshold1 (1-bin_per)   
        Value higher than threshold2 and connected (6-connectivity) to the  
        seeds are set to 1, the other remains 0 
            
        ... and more, see matplotlib    
 
    Returns 
    ------- 
    im_bin: binary image 
    
    References
    ----------
    .. [1] J. Canny. A computational approach to edge detection. IEEE 
    Transactions on Pattern Analysis and Machine Intelligence. 1986; vol. 8, 
    pp.679-698. DOI:10.1109/TPAMI.1986.4767851 
    .. [2] from MATLAB: Threshold estimation (Oliveira et al, 2015) 
     
    Examples 
    -------- 
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, fcrop=(0,10000))   
    
    Convert linear spectrogram into dB
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Smooth the spectrogram
    
    >>> Sxx_dB_blurred = maad.rois.smooth(Sxx_dB)
    
    Detection of the acoustic signature => creation of a mask
    
    >>> im_bin = maad.rois.create_mask(Sxx_dB_blurred, bin_std=1.5, bin_per=0.25, mode='relative') 
    
    Plot spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> fig, (ax1, ax2) = plt.subplots(2, 1)
    >>> maad.util.plot2d(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=10, vmax=70)
    >>> maad.util.plot2d(im_bin, ax=ax2, extent=ext, title='mask)')
    >>> fig.set_size_inches(13,8)
    >>> fig.tight_layout() 
    
    """ 
        
    if mode_bin == 'relative': 
        bin_std=kwargs.pop('bin_std', 6)  
        bin_per=kwargs.pop('bin_per', 0.5)  
        im_bin = _double_threshold_rel(im, bin_std, bin_per, 
                                       verbose, display, savefig, **kwargs) 
         
    elif mode_bin == 'absolute': 
        bin_h=kwargs.pop('bin_h', 0.7)  
        bin_l=kwargs.pop('bin_l', 0.3)  
        im_bin = _double_threshold_abs(im, bin_h, bin_l,
                                       verbose, display, savefig, **kwargs)    
    
    return im_bin  

#%%
def select_rois(im_bin, min_roi=None ,max_roi=None, 
                verbose=False, display=False, savefig = None, **kwargs): 
    """ 
    Select regions of interest based on its dimensions.
    
    The input is a binary mask, and the output is an image with labelled pixels. 
 
    Parameters 
    ---------- 
    im : 2d ndarray of scalars 
        Spectrogram (or image) 
         
    min_roi, max_roi : scalars, optional, default : None 
        Define the minimum and the maximum area possible for an ROI. If None,  
        the minimum ROI area is 1 pixel and the maximum ROI area is the area of  
        the image 
        
    verbose : boolean, optional, default is False
        print messages
         
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
            
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
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
    im_rois: 2d ndarray 
        image with labels as values 
             
    rois: pandas DataFrame 
        Regions of interest with future descriptors will be computed. 
        Array have column names: ``labelID``, ``label``, ``min_y``, ``min_x``,
        ``max_y``, ``max_x``,
        Use the function ``maad.util.format_features`` before using 
        centroid_features to format of the ``rois`` DataFrame 
        correctly.
        
    Examples 
    -------- 
    
    Load audio recording compute the spectrogram in dB.
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, fcrop=(0,20000), display=True)           
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Smooth the spectrogram
    
    >>> Sxx_dB_blurred = maad.sound.smooth(Sxx_dB)
    
     Using image binarization, detect isolated region in the time-frequency domain with high density of energy, i.e. regions of interest (ROIs).
    
    >>> im_bin = maad.rois.create_mask(Sxx_dB_blurred, bin_std=1.5, bin_per=0.5, mode='relative')
    
    Select ROIs from the binary mask.
    
    >>> im_rois, df_rois = maad.rois.select_rois(im_bin, display=True)
    
    We detected the background noise as a ROI, and that multiple ROIs are mixed in a single region. To have better results, it is adviced to preprocess the spectrogram to remove the background noise before creating the mask.
    
    >>> Sxx_noNoise = maad.sound.median_equalizer(Sxx)
    >>> Sxx_noNoise_dB = maad.util.power2dB(Sxx_noNoise)     
    >>> Sxx_noNoise_dB_blurred = maad.sound.smooth(Sxx_noNoise_dB)        
    >>> im_bin2 = maad.rois.create_mask(Sxx_noNoise_dB_blurred, bin_std=6, bin_per=0.5, mode='relative') 
    >>> im_rois2, df_rois2 = maad.rois.select_rois(im_bin2, display=True)
    
    """ 
 
    # test if max_roi and min_roi are defined 
    if max_roi is None:  
        # the maximum ROI is set to the aera of the image 
        max_roi=im_bin.shape[0]*im_bin.shape[1] 
         
    if min_roi is None: 
        # the min ROI area is set to 1 pixel 
        min_roi = 1 
    
    if verbose :
        print(72 * '_') 
        print('Automatic ROIs selection in progress...') 
        print ('**********************************************************') 
        print ('  Min ROI area %d pix² | Max ROI area %d pix²' % (min_roi, max_roi)) 
        print ('**********************************************************') 
 
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
    
    # test if there is a roi
    if len(rois_label)>0 :
        # create a dataframe rois containing the coordonates and the label 
        rois = np.concatenate((np.asarray(rois_label), np.asarray(rois_bbox)), axis=1) 
        rois = pd.DataFrame(rois, columns = ['labelID', 'label', 'min_y','min_x','max_y', 'max_x']) 
        # force type to integer 
        rois = rois.astype({'label': str,'min_y':int,'min_x':int,'max_y':int, 'max_x':int}) 
        # compensate half-open interval of bbox from skimage 
        rois.max_y -= 1 
        rois.max_x -= 1 
        
    else :
        rois = []    
        rois = pd.DataFrame(rois, columns = ['labelID', 'label', 'min_y','min_x','max_y', 'max_x']) 
        rois = rois.astype({'label': str,'min_y':int,'min_x':int,'max_y':int, 'max_x':int}) 
     
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')  
        title  =kwargs.pop('title','Selected ROIs')  
        extent=kwargs.pop('extent',None)
            
        if extent is not None : 
            xlabel = 'frequency [Hz]' 
            figsize=kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
        else: 
            xlabel = 'pseudofrequency [points]'
            figsize=kwargs.pop('figsize',(4, 13)) 
         
        # randcmap = rand_cmap(len(rois_label)) 
        # cmap   =kwargs.pop('cmap',randcmap)  
        cmap   =kwargs.pop('cmap','tab20') 
         
        _, fig = plot2d (im_rois, extent=extent, figsize=figsize,title=title,  
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
            
    return im_rois, rois 
 
#%%
def rois_to_imblobs(im_zeros, rois): 
    """  
    Take a matrix full of zeros and add ones in delimited regions defined by rois.
 
    Parameters 
    ---------- 
    im_zeros : ndarray 
        matrix full of zeros with the size to the image where the rois come from.
     
    rois : DataFrame 
        rois must have the columns names:((min_y, min_x, max_y, max_x) which 
        correspond to the bounding box coordinates 
     
    Returns 
    ------- 
    im_blobs : ndarray 
        matrix with 1 corresponding to the rois and 0 elsewhere 
 
    Examples
    --------
    >>> from maad import rois, util
    >>> import pandas as pd
    >>> import numpy as np
    >>> im_zeros = np.zeros((100,300))
    >>> df_rois = pd.DataFrame({'min_y': [10, 40], 'min_x': [10, 200], 'max_y': [60, 80], 'max_x': [110, 250]})
    >>> im_blobs = rois.rois_to_imblobs(im_zeros, df_rois)
    >>> util.plot2d(im_blobs)
    """ 
    # Check format of the input data 
    if type(rois) is not pd.core.frame.DataFrame : 
        raise TypeError('Rois must be of type pandas DataFrame')   
         
    if not(('min_y' and 'min_x' and 'max_y' and 'max_x')  in rois)  : 
            raise TypeError('Array must be a Pandas DataFrame with column names:((min_y, min_x, max_y, max_x). Check example in documentation.')   
     
    # select the columns 
    rois_bbox = rois[['min_y', 'min_x', 'max_y', 'max_x']] 
    # roi to image blob 
    for min_y, min_x, max_y, max_x in rois_bbox.values: 
        im_zeros[int(min_y):int(max_y+1), int(min_x):int(max_x+1)] = 1 
     
    im_blobs = im_zeros.astype(int) 
     
    return im_blobs 

 
 