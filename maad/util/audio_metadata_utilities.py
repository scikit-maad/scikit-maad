#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to get audio metadata from files

Warning for Windows users. Due to problems using slashes and backslashes, Paths must be raw strings
instead of regular strings, to convert a regular string into a raw string simply add an r before the string.
for example:
    raw string:        r'C:/Users/Documents/Folder/SubFolder/file.wav'
"""

import wave
import pandas as pd
import os
import numpy as np

#%%
def check_file_format(path_audio):
    """
    Check Wave file consistency. Check if WAVE format is correct and if file name 
    follows standard format. The standard format is SITENAME_DATE_TIME.WAV, with 
    DATE as YYYYMMDD and TIME as HHMMSS.

    Parameters
    ----------
    path_audio : str
        Location of audio filename.

    Raises
    ------
    File Not Found
        If file does not exist.

    Returns
    -------
    error : int
        0 if no error is found, 1 if WAVE format is incorrect and 2 if filename has no
        standard format.

    """
    basename = os.path.basename(path_audio)

    # Check Wave format:
    # try to open wav file, if error return only file name and null values on fields
    try:
        with wave.open(path_audio, 'rb') as f:
            _ = f.getparams()

    except FileNotFoundError as fnfe:
        raise fnfe

    except:
        error = 1
        return error

    # Check file name format:
    # 1. File name must have 3 fields separated by underscore '_'
    # 2. Second field (date) must have 8 characters
    # 3. Third field (time) must have 6 charcaters + 4 = 10 ('.WAV')
    if (len(basename.split('_')) != 3):
        error = 2
        return error

    else:
        date_str = basename.split('_')[1]
        time_str = basename.split('_')[2]
        if ((len(date_str) != 8) |  # date_str should have 8 characters
                (len(time_str) != 10) |  # time_str + '.wav' should have 10 characters
                (not (date_str.isnumeric())) |  # date_str should be numeric
                (not (time_str[0:-4].isnumeric()))  # time_str should be numeric
        ):
            error = 2
        else:
            error = 0

    return error

#%% 
def audio_header(path_audio):
    """
    Get audio header information from WAVE file. 
    Header information includes, sample rate, bit depth, number of channels, 
    number of samples, file size and duration.
    
    Parameters
    ----------
    path_audio : str
        Location of audio file.

    Returns
    -------
    metadata : dictionary
        header information.
    
    Examples
    --------
    >>> from maad import util
    >>> dic_metadata = util.audio_header('../data/spinetail.wav')
    >>> print(dic_metadata)
    {'path_audio': '../data/spinetail.wav', 'fname': 'spinetail.wav', 'sample_rate': 44100, 'channels': 1, 'bits': 16, 'samples': 861799, 'fsize': 1723642, 'length': 19.541927437641725}
    """
    basename = os.path.basename(path_audio)

    with wave.open(path_audio, 'rb') as f:
        meta = f.getparams()

    metadata = {'path_audio': path_audio,
                'fname': basename,
                'sample_rate': meta.framerate,
                'channels': meta.nchannels,
                'bits': meta.sampwidth * 8,
                'samples': meta.nframes,
                'fsize': os.path.getsize(path_audio),
                'length': meta.nframes / meta.framerate}
    return metadata

#%% 
def filename_info(path_audio, verbose =False):
    """
    Get information from filename when using standard format. The standard format is
    SITENAME_DATE_TIME.WAV, with DATE as YYYYMMDD and TIME as HHMMSS.

    Parameters
    ----------
    path_audio : str
        Location of audio file.

    Returns
    -------
    metadata : dictionary
        file name information.


    """
    if check_file_format(path_audio) == 0:
        basename = os.path.basename(path_audio)
        date = basename.split("_")[1]
        hour = basename.split("_")[2]
        date_fmt = date[0:4] + "-" + date[4:6] + "-" + date[6:8] + " " + hour[0:2] + ":" + hour[2:4] + ":" + hour[4:6]
        # structure data
        metadata = {'path_audio': path_audio,
                    'fname': basename,
                    'sensor_name': basename.split("_")[0],
                    'date': date_fmt,
                    'time': basename.split("_")[2][0:6]}
    else:
        raise TypeError(
            'File name format not supported. The standard format must be SITENAME_DATE_TIME.WAV, with DATE as YYYYMMDD and TIME as HHMMSS.')
    return metadata

#%%
def get_metadata_file(path_audio, verbose=False):
    """
    Get metadata asociated with audio recordings in audio file. Metadata includes basic 
    information of the audio file format (sample rate, number of channels, bit depth and 
    file size), and date information from the filename. Note however, that this function 
    is intended for use only with audio files with a self-describing header.

    Parameters
    ----------
    path_audio : str
        Path to the audio file name.
    verbose : boolean, optional
        Display error messages. The default is False.

    Returns
    -------
    metadata : dictionary
        Dictionary with metadata.

    """
    path_audio = path_audio.replace('\\', '/')  # for compatibility with Windows

    basename = os.path.basename(path_audio)
    error = check_file_format(path_audio)

    if error != 0:
        metadata = {'path_audio': path_audio,
                    'fname': basename,
                    'sample_rate': np.nan,
                    'channels': np.nan,
                    'bits': np.nan,
                    'samples': np.nan,
                    'fsize': np.nan,
                    'sensor_name': np.nan,
                    'date': np.nan,
                    'time': np.nan,
                    'length': np.nan}
        if verbose:
            print('Incorrect name or wave format. Return null values for: ', path_audio)
        return metadata


    # Execute function if no error
    else:
        info_header = audio_header(path_audio)
        info_fname = filename_info(path_audio)
        metadata = {'path_audio': path_audio,
                    'fname': info_header['fname'],
                    'sample_rate': info_header['sample_rate'],
                    'channels': info_header['channels'],
                    'bits': info_header['bits'],
                    'samples': info_header['samples'],
                    'length': info_header['length'],
                    'fsize': info_header['fsize'],
                    'sensor_name': info_fname['sensor_name'],
                    'date': info_fname['date'],
                    'time': info_fname['time']}
    return metadata

# %%
def get_metadata_dir(path_dir, verbose=False):
    """
    Get metadata asociated with audio recordings in a directory. Metadata includes basic 
    information of the audio file format (sample rate, number of channels, bit depth and 
    file size), and date information from the filename. Note however, that this function 
    is intended for use only with audio files with a self-describing header.

    Parameters
    ----------
    path_dir : str
        Path of either a directory or a file. it will select all wav files in the parent folder
        (of either the file or directory in path_dir). The search for file is performed recursively.
    verbose : boolean, optional
        Output file progress. The default is False.

    Returns
    -------
    df_metadata : pandas.DataFrame
        Dataframe with metadata, files as rows and metadata as columns.
    
    See Also
    --------
    maad.util.get_metadata_file, maad.util.audio_header, maad.util.filename_info
    
    Examples
    --------
    >>> from maad import util
    >>> df_metadata = util.get_metadata_dir('../data/indices/')
    """

    # List all files recursively and select only wav files.
    root = os.path.dirname(path_dir)  # if filename is given, look for the directory.
    flist = [os.path.join(path_dir, name) for path_dir, subdirs, files in os.walk(root) for name in files]
    flist_wav = [k for k in flist if '.WAV' in k] + [k for k in flist if '.wav' in k]

    # Get metadata for each file
    df_metadata = pd.DataFrame()
    for count, file in enumerate(flist_wav):
        if verbose:
            print(count, '/', len(flist_wav), ':', os.path.basename(file))

        data = get_metadata_file(file, verbose)
        df_metadata = pd.concat([df_metadata, pd.DataFrame.from_records([data])])

    df_metadata.reset_index(drop=True, inplace=True)
    
    return df_metadata



#%% Examples of use
"""
Test functions 


# 1. Read metadata from a valid file with correct name format -> read ok
path_audio = './test_data/S4A03895_20190522_000000.wav'
get_metadata_file(path_audio)
audio_header(path_audio)
filename_info(path_audio)
check_file_format(path_audio)  # should be 0

# 2. Read metadata from a valid file with incorrect name format -> null output
path_audio = './test_data/spinetail_20220219_30222L.wav'
get_metadata_file(path_audio)
audio_header(path_audio)
filename_info(path_audio)
check_file_format(path_audio) # should be error=2

# 3. Read metadata from a wav file with incorrect format and correct name format -> null output
path_audio = './test_data/NOHEADER_20190522_000000.wav'
get_metadata_file(path_audio)
audio_header(path_audio)
filename_info(path_audio)
check_file_format(path_audio)  # should be error=1

# 4. Read metadata from a file with no header and incorrect name format -> null output
path_audio = './test_data/NOHEADER_BAD_FNAME_FORMAT.wav'
get_metadata_file(path_audio)
audio_header(path_audio)
filename_info(path_audio)
check_file_format(path_audio)  # should be error=1

"""


