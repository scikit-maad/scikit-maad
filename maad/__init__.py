# -*- coding: utf-8 -*-
""" Multiresolution Analysis of Acoustic Diversity Scikit (Toolbox for SciPy)

scikit-maad is a modular toolbox to analyze ecoacoustics datasets in Python 3. 
This package was designed to bring flexibility to find regions of interest,
and to compute acoustic features in audio recordings. This workflow opens 
the possibility to use powerfull machine learning algorithms through 
scikit-learn, allowing to identify key patterns in all kind of soundscapes.

Subpackages
-----------

acoustics
    Collection of functions to describe the physics of acoustic waves such as
    sound pressure level, attenuation laws or propagation - :ref:`maad.acoustics`

features
    Compute descriptors to characterize sounds - :ref:`maad.features`

rois
    Find regions of interest in 1D and 2D signals - :ref:`maad.rois`

sound
    Load, preprocess and transform (e.g. stft) audio signals - :ref:`maad.sound`
    
util
    Miscelaneous and useful set of tools used in the audio analysis framework - :ref:`maad.util`    
"""
from .version import __version__

from . import spl
from . import features
from . import rois
from . import sound
from . import util
