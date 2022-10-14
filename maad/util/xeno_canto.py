#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Collection of functions to send queries to www.xeno-canto.org, get dataframe
with all xeno-canto fields and eventually download bird sound files with
JSON metadata.

"""   
#
# Authors:  original author Karoliina Oksanen, 2014
#           Updated to python 3.7.4, Agnieszka Mikolajczyk, 2019
#           Modified for scikit-maad by Sylvain HAUPERT, 2021        
#
# License: New BSD License


import urllib.request
import json
import pandas as pd
from pathlib import Path
import numpy as np
import os

# %%
def xc_query(searchTerms,
             max_nb_files = None,
             format_time = False,
             format_date = False,
             random_seed = 1979,
             verbose=False):
    """
    Query metadata from Xeno-Canto website depending on the search terms. The
    audio recordings metadata are grouped and stored in a dataframe.

    Parameters
    ----------
    searchTerms : list
        list of search terms to perform the query
        The main seach terms are :
        - grp : birds
        - gen : genus
        - ssp : subspecies
        - en  : english name
        - q   : quality
        - cnt : country
        - len : length
        - area : continent (europe, africa, america, asia)
        see more here : https://www.xeno-canto.org/help/search
    max_nb_files: integer, optional
        Maximum number of audio files requested. The default is None
    format_time : boolean, optional
        Time in Xeno-Canto is not always present neither correctly formated. 
        If true, time will be correctly formated to be processed as DateTime 
        format. When formating is not possible, the row is dropped. 
        The default is False
    format_date : boolean, optional
        Date in Xeno-Canto is not always present neither correctly formated. 
        If true, rows with uncorrect format of date are dropped.
    random_seed : integer, optional
        Fix the random seed in order to get the same result every time the 
        function is called
    verbose : boolean, optional
        Print messages during the execution of the function. The default is False.

    Returns
    -------
    df_dataset : pandas DataFrame
        Dataframe containing all the recordings metadata matching search terms
    """
    numPages = 1
    page = 1
    df_dataset = pd.DataFrame()
    while page < numPages+1:
        if verbose:
            print("Loading page "+str(page)+"...")
        url = 'https://www.xeno-canto.org/api/2/recordings?query={0}&page={1}'.format(
            '%20'.join(searchTerms), page)
        if verbose:
            print(url)
        jsonPage = urllib.request.urlopen(url)
        jsondata = json.loads(jsonPage.read().decode('utf-8'))
        # check number of pages
        numPages = jsondata['numPages']
        # Append pandas dataframe of records & convert to .csv file
        df_dataset = df_dataset.append(pd.DataFrame(jsondata['recordings']))
        # increment the current page
        page = page+1
        
    # if no limit in the number of files
    if max_nb_files is not None :
        # test if the number of files is greater than the maximum number of
        # resquested files
        if len(df_dataset) > max_nb_files : 
            df_dataset = df_dataset.sample(n = max_nb_files,
                                           random_state = random_seed)
        
    if verbose:
        print("Found", numPages, "pages in total.")
        print("Saved metadata for", len(df_dataset), "files")
    
    # test if the dataset is not empty
    if len(df_dataset)>0:
        # rearrange index to be sure to have unique and increasing index
        df_dataset.reset_index(drop=True, inplace=True)
        # the format of length is not correct (missing 0 before 0:45 => 00:45)
        # Correct the format of length for length shorten than 9:59 (4 characters)
        # by adding a 0
        df_dataset['length'].where(~(df_dataset.length.str.len()==4), 
                                    other='0'+ df_dataset[df_dataset.length.str.len()==4].length, 
                                    inplace=True) 
        if format_time == True :
            # rearrange index to be sure to have unique and increasing index
            df_dataset.reset_index(drop=True, inplace=True)
            # the format of time is not always correct
            # replace . by :
            df_dataset['time'].replace(to_replace = '[.]', value=':', regex= True)
            df_dataset['time'].replace(to_replace = '[ ] ', value='', regex= True)
            # drop rows where there is no valid time information that can be corrected
            df_dataset = df_dataset[(df_dataset.time.str.match('^(0[0-9]|1[0-9]|2[0-3])[:]([0-5][0-9])$')) | 
                                    (df_dataset.time.str.match('^([0-9])[:]([0-5][0-9])$'))]
            # Correct the format of time when 0 is missing (missing 0 before 0:45 => 00:45)
            # by adding a 0
            df_dataset['time'][df_dataset.time.str.match('^([0-9])[:]([0-5][0-9])$')] = '0' + df_dataset[df_dataset.time.str.match('^([0-9])[:]([0-5][0-9])$')].time

            if verbose:
                print("Keeped metadata for", len(df_dataset), "files after formating time")
            
        if format_date == True :
            # rearrange index to be sure to have unique and increasing index
            df_dataset.reset_index(drop=True, inplace=True)
            # drop rows where there is no valid date information
            df_dataset = df_dataset[df_dataset.date.str.match(r'^(20[0-9][0-9]|19[0-9][0-9])-(0[1-9]|1[0-2])-([1-9]|1[0-9]|2[0-9]|3[0-1])$')]
            
            if verbose:
                print("Keeped metadata for", len(df_dataset), "files after formating date")
        
        if (format_time == True) and (format_date == True) :         
            # add a column with the week number
            df_dataset['week'] = pd.to_datetime(df_dataset['date']).dt.isocalendar()['week']
            # add a column with datetime in DateTime format
            df_dataset['datetime'] =  pd.to_datetime(df_dataset['time']+' '+ df_dataset['date'])
            
    # rearrange index to be sure to have unique and increasing index
    df_dataset.reset_index(drop=True, inplace=True)

    return df_dataset

# %%
def xc_multi_query(df_query,
                   max_nb_files = None,
                   format_time = False,
                   format_date = False,
                   random_seed = 1979,
                   verbose = False):
    """
    Multi_query performs multiple queries following the search terms defined
    in the input dataframe

    Parameters
    ----------
    df_query : pandas DataFrame
        Dataframe with search terms. Each row corresponds to a new query. 
        Columns corresponds to the search terms allowed by Xeno-Canto
    max_nb_files: integer, optional
        Maximum number of audio files requested. The default is None
    format_time : boolean, optional
        Time in Xeno-Canto is not always present neither correctly formated. 
        If true, time will be correctly formated to be processed as DateTime 
        format. When formating is not possible, the row is dropped. 
        The default is False
    format_date : boolean, optional
        Date in Xeno-Canto is not always present neither correctly formated. 
        If true, rows with uncorrect format of date are dropped.
    random_seed : integer, optional
        Fix the random seed in order to get the same result every time the 
        function is called
    verbose : boolean, optional
        Print messages during the execution of the function. The default is False.

    Returns
    -------
    df_dataset : pandas DataFrame
        Dataframe containing all the recordings metadata matching 
        the search terms.

    """

    df_dataset = pd.DataFrame()
    for index, row in df_query.iterrows():
        searchTerms = row.tolist()
        df_dataset = df_dataset.append(xc_query(searchTerms, 
                                                max_nb_files,
                                                format_time,
                                                format_date,
                                                random_seed,
                                                verbose))

    # rearrange index to be sure to have unique and increasing index
    df_dataset.reset_index(drop=True, inplace=True)
    
    return df_dataset

#%%
def xc_selection(df_dataset,
                 max_nb_files=100,
                 max_length='01:00',
                 min_length='00:10',
                 min_quality='B',
                 verbose = False):
    """
    Select a maximum number of recordings depending on their quality and 
    duration in order to create an homogeneous dataset.

    Parameters
    ----------
    df_dataset : pandas DataFrame
        Dataframe containing all the recordings metadata 
    max_nb_files : int, optional
        Max number of audio files per species. The default is 100.
    max_length : string, optional
        Max duration of the audio files. The default is '01:00'.
    min_length : string, optional
        Min duration of the audio files. The default is '00:10'.
    min_quality : string, optional
        Min quality of the audio files. The default is 'B'.
    verbose : boolean, optional
        Print messages during the execution of the function. The default is False.

    Returns
    -------
    df_dataset_out : pandas DataFrame
        Dataframe containing the selected recordings metadata 

    """

    df_dataset_out = pd.DataFrame()
    quality = ['A', 'B', 'C', 'D', 'E']
    if min_quality == 'A' :
        quality =['A']
    elif min_quality == 'B' :
        quality = ['A', 'B']
    elif min_quality == 'C':
        quality = ['A', 'B', 'C']
    elif min_quality == 'D':
        quality = ['A', 'B', 'C', 'D']
    unique_species = pd.unique(df_dataset.gen + ' ' + df_dataset.sp)
    for name in unique_species:
        if verbose : print(name)
        # extract the genus and species from the scientific name
        gen = name.rpartition(' ')[0]
        sp = name.rpartition(' ')[2]
        # select the corresponding to the species
        # !! the string test is case sensitive (the genus start with a upper case)
        subdf_dataset = df_dataset[((df_dataset.gen == gen) &
                                    (df_dataset.sp == sp))]
        # sort the dataframe corresponding to the species by audio quality
        subdf_dataset = subdf_dataset.sort_values(by='q')

        # Counter initialization
        current_nb_files = 0
        requested_nb_files = 0
        current_quality = 0
        while (current_nb_files < max_nb_files) & (current_quality < len(quality)):
            requested_nb_files = max_nb_files - current_nb_files
            q = quality[current_quality]
            if verbose : print('    ... request %2.0f files of quality %s' %
                  (requested_nb_files, q))
            
            mask1 = ((subdf_dataset.q == q) &
                    (subdf_dataset.length <= max_length) &
                    (subdf_dataset.length >= min_length))

            if len(subdf_dataset[mask1]) >= requested_nb_files:
                # create a temp dataframe with the selected rows
                df_temp = subdf_dataset[mask1].sort_values(
                    by='length', ascending=False).iloc[0:requested_nb_files]
                # add the rows to the output dataframe
                df_dataset_out = df_dataset_out.append(df_temp)
                # drop the selected rows to avoid future selection
                subdf_dataset.drop(df_temp.index, axis=0, inplace=True)
            else :  
                # create a temp dataframe with the selected rows
                df_temp = subdf_dataset[mask1].sort_values(
                    by='length', ascending=False)
                # add the rows to the output dataframe
                df_dataset_out = df_dataset_out.append(df_temp)
                # drop the selected rows to avoid future selection
                subdf_dataset.drop(df_temp.index, axis=0, inplace=True)
            if verbose : 
                print('    --> found %2.0f files of quality %s and %s<length<%s'%
                     (len(df_temp), q, min_length, max_length ))
            current_nb_files += len(df_temp)
            requested_nb_files = max_nb_files - current_nb_files
            current_quality += 1

        if verbose : print('    total files : %2.f' %current_nb_files)
        if verbose : print("-----------------------------------------")
        
    return df_dataset_out

#%%
def xc_download(df,
                 rootdir,
                 dataset_name='dataset',
                 overwrite = False,
                 save_csv=False,
                 verbose = False):
    """
    Download the audio files from Xeno-Canto based on the input dataframe
    It will create directories for each species if needed

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing the selected recordings metadata 
    rootdir : string
        Path to the directory where the whole dataset will be saved
    dataset_name : string, optional
        Name of the dataset that will be created as a parent directory . 
        The default is 'dataset'.
    overwrite : boolean, optional
        Test if the directory where the audio files will be downloaded already
        exists. if True, it will download the data in the directory anyway.
        Otherwise, if False, it will not download audio files.
    save_csv : boolean, optional
        if True, the csv corresponding to the species will be saved in the
        directory of the species. The default is False.
    verbose : boolean, optional
        Print messages during the execution of the function. The default is False.

    Returns
    -------
    fullpath_list : string
        Returns a list with all the paths to the newly downloaded audio files
    """
    # Check whether the specified path is an existing directory or not 
    isdir = os.path.isdir(rootdir) 
    
    # Download audio
    fullpath_list = []
    if (isdir == False) or ((isdir == True) and (overwrite == True)) : 
        if (overwrite == True):
            if verbose:
                print(
                    "The directory "
                    + rootdir
                    + " already exists and will be overwritten" )

        # rearrange index to be sure to have unique and increasing index
        df['index'] = np.arange(0, len(df))
        df.set_index(['index'], inplace=True)
        
        if verbose :
            numfiles = len(df)
            print("A total of", numfiles, "files will be downloaded")
            
        # change type of rootdir into Path
        rootdir = Path(rootdir)
        for index, row in df.iterrows():
            # create a name for the directory
            name_dir = row.gen + ' ' + row.sp + '_' + row.en
            # create data/xeno-canto-dataset directory
            path = rootdir / dataset_name / name_dir
            if not os.path.exists(path):
                if verbose :
                    print("Creating subdirectory " + 
                          str(path) + 
                          " for downloaded files...")
                os.makedirs(path)
            # get filenames
            filename = 'XC' + row.id + '.mp3'
            # get website recording http download address 
            fileaddress = row.file
            if verbose : 
                print("Saving file ", index+1, "/", numfiles, ": " + fileaddress)
            fullpath, _ = urllib.request.urlretrieve(fileaddress, path / filename)
            fullpath_list += [str(fullpath)]
            # save csv
            if save_csv:
                filename_csv = str(path/'metadata.csv')
                # test if the csv file doesn't exit
                if os.path.exists(filename_csv) == False:
                    mask = (df.gen == row.gen) & (df.sp == row.sp)
                    df[mask].to_csv(filename_csv, index=False)    
    else :
        if verbose:
            print(
                "The directory "
                + rootdir
                + " already exists"
            )
            
    return fullpath_list

# %%
if __name__ == '__main__':
    
    df_query = pd.DataFrame()
    df_species = pd.DataFrame()
    
    # species
    df_species['scientific name'] = ['Agelaius phoeniceus',
                                     'psittacula krameri',
                                     'Ardea herodias']
    # query
    gen = []
    sp = []
    for name in df_species['scientific name']:
        gen.append(name.rpartition(' ')[0])
        sp.append(name.rpartition(' ')[2])

    df_query['gen'] = gen
    df_query['sp'] = sp
    df_query['q'] = 'q:A'
    df_query['len'] = 'len:"10-60"'

    df_dataset = xc_multi_query(df_query, 
                                max_nb_files = 5,
                                verbose=True)
    
    print('number of files in the dataset is %d'%(len(df_dataset)))
    print(df_dataset.head(15))
    
    # df_dataset.to_csv('dataset_xc.csv')
    
    # xc_download(df_dataset,'my_dataset', , save=True)
