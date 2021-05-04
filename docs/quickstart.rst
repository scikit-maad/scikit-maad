Quickstart
^^^^^^^^^^

The full package is imported as ``maad``::

    import maad
    
All functions are organized within modules. The module :any:`maad.sound` has functions to load and preprocess audio signals, :any:`maad.rois` provides tools to find regions of interest in audio (**1D**) and spectrogram signals (**2D**), :any:`maad.features` include functions to compute robust descriptors to characterize audio signals, and :any:`maad.util` has a handfull of useful set of tools used in the audio analysis framework.

To load submodules juste type::

    from maad import sound, rois
    
To use scikit-maad tools, audio must be loaded as a numpy array. The function :py:func:`maad.sound.load` is a simple and effective way to load audio from disk. For example, download the spinetail example to your working directory (`link <https://github.com/scikit-maad/scikit-maad/blob/production/data/spinetail.wav>`_) and type::

    s, fs = sound.load('spinetail.wav')
    
You can then apply any analysis to find regions of interest or characterize your audio signals.::
    
    rois.find_rois_cwt(s, fs, flims=(4500,8000), tlen=2, th=0, display=True)

We provide a diversified audio dataset in our `repository <https://scikit-maad.github.io/scikit-maad/>`_ to test the package functions. For more information, visit the :ref:`modindex` and the :ref:`Example gallery`.
