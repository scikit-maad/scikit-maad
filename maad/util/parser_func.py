#!/usr/bin/env python
""" Utilitary functions to parse or read files"""
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
import pandas as pd
import os
from datetime import datetime

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
    tab_out = pd.concat([t_info['label'].astype('str'), 
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
        This object contains the date of creation of the file extracted from
        the filename postfix. 
        The filename must follow this format :
            XXXX_yyyymmdd_hhmmss.wav
            with yyyy : year / mm : month / dd: day / hh : hour (24hours) /
            mm : minutes / ss : seconds
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

def date_parser (datadir, verbose=False):
    c_file = []
    c_date = []
    # find a file in subdirectories
    for root, subFolders, files in os.walk(datadir):
        for count, file in enumerate(files):
            if verbose:print(file)
            if '.wav' in file:
                filename = os.path.join(root, file)
                c_date.append(date_from_filename(filename))    
                c_file.append(filename)               
                
    ####### SORTED BY DATE
    # create a Pandas dataframe with date as index
    df = pd.DataFrame({'file':c_file},index=c_date)
    # sort dataframe by date
    df = df.sort_index(axis=0)
    return df