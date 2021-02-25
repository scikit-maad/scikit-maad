#!/usr/bin/env python
""" Utilitary functions to parse and read audio and text files. """
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License

#%%
# =============================================================================
# Load the modules
# =============================================================================
# Import external modules
import numpy as np 
import pandas as pd
import os
from datetime import datetime
from pathlib import Path # in order to be Windows/linux/MacOS compatible


#%%
# =============================================================================
# Private functions
# =============================================================================
def _date_from_filename (filename):
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
        This object contains the date of creation of the file extracted from
        the filename postfix. 
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


#%%
# =============================================================================
# Public functions
# =============================================================================
def read_audacity_annot (audacity_filename):
    """
    Read audacity annotations file (or labeling file) and return a Pandas Dataframe
    with the bounding box and the label of each region of interest (ROI).
    
    Parameters
    ----------
    audacity_filename : String
        Path to the audacity file

    Returns
    -------
    tab_out : Pandas Dataframe 
        Region of interest with time-frequency limits and manual annotation label
    
    References
    ----------
    https://manual.audacityteam.org/man/label_tracks.html  
    
    Examples
    --------
    >>> from maad import sound
    >>> from maad.util import power2dB, read_audacity_annot, format_features, overlay_rois
    >>> s, fs = sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=1024//2)
    >>> Sxx_db = power2dB(Sxx_power) + 96
    >>> df_rois = read_audacity_annot('../data/cold_forest_daylight_label.txt') 
    >>> df_rois = format_features(df_rois, tn, fn)
    >>> overlay_rois(Sxx_db, df_rois, **{'vmin':0,'vmax':96,'extent':ext})
    
    """
    # read file with tab delimiter
    tab_in = pd.read_csv(audacity_filename, delimiter='\t', header=None)
    
    # arrange data
    t_info = tab_in.loc[np.arange(0,len(tab_in),2),:]
    t_info = t_info.rename(index=str, columns={0: 'min_t', 1: 'max_t', 2:'label'})
    t_info = t_info.reset_index(drop=True)
    
    f_info = tab_in.loc[np.arange(1,len(tab_in)+1,2),:]
    f_info = f_info.rename(index=str, columns={0: 'slash', 1: 'min_f', 2:'max_f'})
    f_info = f_info.reset_index(drop=True)
    
    # return dataframe
    tab_out = pd.concat([t_info['label'].astype('str'), 
                         t_info['min_t'].astype('float32'), 
                         f_info['min_f'].astype('float32'), 
                         t_info['max_t'].astype('float32'), 
                         f_info['max_f'].astype('float32')],  axis=1)

    return tab_out

#%%

def write_audacity_annot(fname, df_rois):
    """ 
    Write audio segmentation to file (Audacity format).
    
    Parameters
    ----------
    fname: str
        filename to save the segmentation
    df_rois: pandas dataframe
        Dataframe containing the coordinates corresponding to sound signatures
        In case of only temporal annotations : df_rois must contain at least
        the columns 'mint_t', 'max_t' 
        In case of bounding box (temporal eand frequency limits) :: df_rois 
        must contain at least the columns 'min_t', 'max_t', 'min_f', 'max_f'
            
    Returns
    -------
    df_to_save
        Dataframe that has been saved
    
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx_power, tn, fn, ext = maad.sound.spectrogram(s, fs)
    >>> Sxx_db = maad.util.power2dB(Sxx_power) + 96
    >>> Sxx_power_noNoise= maad.sound.median_equalizer(Sxx_power)
    >>> Sxx_db_noNoise = maad.util.power2dB(Sxx_power_noNoise)
    >>> Sxx_db_noNoise_smooth = maad.sound.smooth(Sxx_db_noNoise, std=0.5)
    >>> im_mask = maad.rois.create_mask(im=Sxx_db_noNoise_smooth, mode_bin ='relative', 
                                        bin_std=8, bin_per=0.5)
    >>> im_rois, df_rois = maad.rois.select_rois(im_mask, min_roi=25, max_roi=None)
    >>> df_rois = maad.util.format_features(df_rois, tn, fn)
    
    Change path to save the file containing the labels position
    
    >>> df_to_save = maad.util.write_audacity_annot('save.txt', df_rois)
    
    Import the wav file then the label file in Audacity
    
    """
    if df_rois.size==0:
        print(fname, '> No detection found')
        df = pd.DataFrame(data=None)
        df.to_csv(fname, sep=',',header=False, index=False)
    else:
        # if there is no label, create a vector with incremental values
        if 'label' not in df_rois :
            label = np.arange(0,len(df_rois))
        # if no frequency coordinates, only temporal annotations
        if ('min_f' not in df_rois) or ('max_f' not in df_rois) :
            df_to_save = pd.DataFrame({'min_t':df_rois.min_t, 
                                    'max_t':df_rois.max_t, 
                                    'label':label})
        elif ('min_f' in df_rois) and ('max_f'  in df_rois) :
            df_to_save_odd = pd.DataFrame({'index': np.arange(0,len(df_rois)*2,2),
                                        'min_t':df_rois.min_t, 
                                        'max_t':df_rois.max_t, 
                                        'label':df_rois.label})
            df_to_save_even = pd.DataFrame({'index': np.arange(1,len(df_rois)*2,2),
                                         'min_t':'\\', 
                                         'max_t':df_rois.min_f, 
                                         'label':df_rois.max_f})
            df_to_save = pd.concat([df_to_save_odd,df_to_save_even])
            df_to_save = df_to_save.set_index('index')
            df_to_save = df_to_save.sort_index()
        df_to_save.to_csv(fname, index=False, header=False, sep='\t') 
    
    return df_to_save

#%%

def date_parser (datadir, dateformat ="SM4", extension ='.wav', verbose=False):
    """
    Parse all filenames contained in a directory and its subdirectories.
    
    Keeps only filenames corresponding to extension.
    Filenames must follow :
        - SM4 format (XXXX_yyyymmdd_hhmmss.wav) 
        - or POSIX format (for audiomoth)  
    
    The result is a panda dataframe with 'Date' as index and 'File' (with full path as column) 
    
    Parameters
    ----------
    filename : string
        filename must follow 
        - SM4 format : XXXX_yyyymmdd_hhmmss.wav, ex: S4A03895_20190522_191500.wav
        - Audiomoth format : 8 Hexa, ex: 5EE84AC8.wav
            
    Returns
    -------
    df : Pandas dataframe
        This dataframe has one column
        - 'file' => full path + filename
        and a index 'Date' with the type Datetime
    
    Examples
    --------
    >>> df = maad.util.date_parser("../data/indices/", dateformat='SM4', verbose=True)
    >>> list(df)
    >>> df
                                                                 file
    Date                                                             
    2019-05-22 00:00:00  ../data/indices/S4A03895_20190522_000000.wav
    2019-05-22 00:15:00  ../data/indices/S4A03895_20190522_001500.wav
    2019-05-22 00:30:00  ../data/indices/S4A03895_20190522_003000.wav
    2019-05-22 00:45:00  ../data/indices/S4A03895_20190522_004500.wav
    2019-05-22 01:00:00  ../data/indices/S4A03895_20190522_010000.wav
    2019-05-22 01:15:00  ../data/indices/S4A03895_20190522_011500.wav
    2019-05-22 01:30:00  ../data/indices/S4A03895_20190522_013000.wav
    2019-05-22 01:45:00  ../data/indices/S4A03895_20190522_014500.wav
    2019-05-22 02:00:00  ../data/indices/S4A03895_20190522_020000.wav
    ...
    """
    
    c_file = []
    c_date = []
    # find a file in subdirectories
    for root, subFolders, files in os.walk(datadir):
        for count, file in enumerate(files):
            if verbose: print(file)
            if extension.upper() in file or extension.lower() in file :
                filename = os.path.join(root, file)
                c_file.append(filename) 
                if dateformat == "SM4":
                    c_date.append(_date_from_filename(filename))      
                elif dateformat == "POSIX" :
                    file_stem = Path(filename).stem
                    print(file_stem)
                    posix_time = int(file_stem, 16)
                    dd = datetime.utcfromtimestamp(posix_time).strftime('%Y-%m-%d %H:%M:%S')
                    print(dd)
                    c_date.append(dd)                          
                
    ####### SORTED BY DATE
    # create a Pandas dataframe with date as index
    df = pd.DataFrame({'file':c_file, 'Date':c_date})
    # define Date as index
    df.set_index('Date', inplace=True)
    # sort dataframe by date
    df = df.sort_index(axis=0)
    return df