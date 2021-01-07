Quickstart
^^^^^^^^^^

**scikit-maad** is a free, open-source and modular Python package to analyze 
ecoacoustics datasets.  The package is imported as ``maad``::

    import maad
    
The functions are found within modules. The module :ref:`sound` has functions to load and preprocess audio signals, :ref:`rois` provides tools to find regions of interest in audio (1D) and spectrogram signals (2D), :ref:`features` include functions to compute robust descriptors to characterize audio signals, and :ref:`util` has a handfull of useful set of tools used in the audio analysis framework.

To load a submodule juste type::

    from maad import sound, rois
    
The package provides some example audio files to test the functions and utilities provided. Audio files can be loaded with the function ``maad.sound.load``::

    s, fs = sound.load('./data/spinetail.wav')
    
You can then apply any analysis to find regions of interest or characterize your audio signals.::
    
    rois.find_rois_cwt(s, fs, flims=(4500,8000), tlen=2, th=0, display=True)

For more information, visit the :ref:`modindex` and the :ref:`Example gallery`.
