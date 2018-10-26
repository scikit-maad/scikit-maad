# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:59:44 2018

This script gives an example of how to use scikit-MAAD package

@author: haupert
"""

print(__doc__)

# Clear all the variables 
from IPython import get_ipython
get_ipython().magic('reset -sf')
 
# =============================================================================
# Load the modules
# =============================================================================

import matplotlib.pyplot as plt
import pandas as pd # for csv
import os

############## Import MAAD modules
# Test the current operating system
from pathlib import Path, PureWindowsPath # in order to be wind/linux/MacOS compatible
if os.name =='nt':
    # Path
    root_maad_path = Path("D:\\")
elif os.name =='posix':
    # LINUX or MACOS
    root_maad_path = Path("/home/haupert/DATA")
else:
    print ("WARNING : operating system unknow")    
maad_path = root_maad_path/Path('mes_projets/_TOOLBOX/Python/maad_project/scikit-maad')
os.sys.path.append(maad_path.as_posix())
import maad
#####################################

# change the path to the current path where the script is located
# Get the current dir of the current file
dir_path = os.path.dirname(os.path.realpath('__file__'))
os.chdir(dir_path)

# Close all the figures (like in Matlab)
plt.close("all")

"""****************************************************************************
# -------------------          options              ---------------------------
****************************************************************************"""
# root directory of the files

# Test the current operating system
if os.name =='nt':
    # WINDOWS
    root_data_path = Path("D:\\")
    #root_data_path = PureWindowsPath("F:\\")
    root_save_path = Path("D:\\")   
elif os.name =='posix':
    # LINUX or MACOS
    root_data_path = Path("/home/haupert/DATA")
else:
    print ("WARNING : operating system unknow")

datadir=root_data_path/Path('mes_projets_data/FRANCE/PNR_JURA/MAGNETO_01_S4A03895/Data')
         
"""****************************************************************************
# -------------------          end options          ---------------------------
****************************************************************************"""

df = maad.util.date_parser(datadir)

##### EXAMPLES
# Returning an array containing the hours for each row in your dataframe
df.index.hour
# grab all rows where the time is between 12h and 13h,
df.between_time('05:00:00','05:30:00')
# Increment the time by 1 microsecond
df.index = df.index+ pd.Timedelta(microseconds=1) 


from scipy.io import wavfile 

fullfilename = df.between_time('05:00:00','08:00:00')['file'][0]

for fullfilename in df.between_time('05:00:00','08:00:00')['file']:
    fs, s = wavfile.read(fullfilename)
