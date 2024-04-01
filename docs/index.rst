.. scikit-maad documentation master file, created by
   sphinx-quickstart on Tue Aug 18 11:29:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: logo/maad_key_visual_black.png
   :align: center


Soundscape analysis in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**scikit-maad** is an open source Python package dedicated to the quantitative analysis of environmental audio recordings. This package was designed to (1) load and process digital audio, (2) segment and find regions of interest, (3) compute acoustic features, and (4) estimate sound pressure levels. This workflow opens the possibility to scan large audio datasets and use powerful machine learning techniques, allowing to measure acoustic properties and identify key patterns in all kinds of soundscapes.

.. note::
  The latest stable release of scikit-maad is now version 1.4.1. Explore the enhancements on `GitHub <https://github.com/scikit-maad/scikit-maad/releases/>`_.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install
   quickstart
   audio_dataset
   
.. toctree::
   :caption: Documentation
   :maxdepth: 1
   
   sound
   rois
   features
   spl
   util
   
.. toctree::
   :maxdepth: 2
   :caption: Example Gallery

   _auto_examples/index
   

Citing scikit-maad
~~~~~~~~~~~~~~~~~~

.. note::
  If you find this package useful in your research, we would appreciate citations to:
    
  Ulloa, J. S., Haupert, S., Latorre, J. F., Aubin, T., & Sueur, J. (2021). scikit-maad: An open-source and modular toolbox for quantitative soundscape analysis in Python. Methods in Ecology and Evolution, 2041-210X.13711. https://doi.org/10.1111/2041-210X.13711


About the project
~~~~~~~~~~~~~~~~~

In 2018, we began to translate a set of audio processing functions from Matlab to an open-source programming language, namely, Python. These functions provided the necessary tools to replicate the Multiresolution Analysis of Acoustic Diversity (MAAD), a method to estimate animal acoustic diversity using unsupervised learning (`Ulloa et al., 2018 <https://doi.org/10.1016/j.ecolind.2018.03.026>`_). We soon realized that Python provided a suitable environment to extend these core functions and to develop a flexible toolbox for our research. During the past few years, we added over 50 acoustic indices, plus a module to estimate the sound pressure level of audio events. Furthermore, we updated, organized, and fully documented the code to make this development accessible to a much wider audience. This work was initiated by `Juan Sebastian Ulloa <https://www.researchgate.net/profile/Juan_Ulloa>`_, supervised by **Jérôme Sueur** and **Thierry Aubin** at the `Muséum National d'Histoire Naturelle <http://isyeb.mnhn.fr/fr>`_ and the `Université Paris Saclay <http://neuro-psi.cnrs.fr/>`_ respectively. Python functions have been added by `Sylvain Haupert <https://www.researchgate.net/profile/Sylvain_Haupert>`_, `Juan Felipe Latorre <https://www.researchgate.net/profile/Juan_Latorre_Gil>`_ (`Universidad Nacional de Colombia <https://unal.edu.co/>`_) and Juan Sebastián Ulloa (`Instituto de Investigación de Recursos Biológicos Alexander von Humboldt <http://www.humboldt.org.co/>`_).

Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`