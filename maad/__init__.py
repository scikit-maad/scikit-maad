# -*- coding: utf-8 -*-
""" Multiresolution Analysis of Acoustic Diversity Scikit (Toolbox for SciPy)

scikit-maad is a modular toolbox to analyze ecoacoustics datasets in Python 3. 
This package was designed to bring flexibility to find regions of interest,
and to compute acoustic features in audio recordings. This workflow opens 
the possibility to use powerfull machine learning algorithms through 
scikit-learn, allowing to identify key patterns in all kind of soundscapes.

Subpackages
-----------
sound
    Load and transform (e.g. stft) audio signals

rois
    Find regions of interest in 1D and 2D signals

features
    Compute descriptors to characterize sounds
    
cluster
    Cluster regions of interest using High Dimensional Data Clsutering (HDDC)

util
    Miscelaneous and useful set of tools used in the audio analysis framework
"""

from . import sound
from . import rois
from . import features
from . import cluster
from . import util
