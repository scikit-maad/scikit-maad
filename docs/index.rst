.. scikit-maad documentation master file, created by
   sphinx-quickstart on Tue Aug 18 11:29:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: logo/maad_key_visual_black.png
   :align: center


Soundscape analysis in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

scikit-maad is an open source Python package dedicated to the quantitative analysis of environmental audio recordings. This package was designed to (1) load and process digital audio, (2) segment and find regions of interest, (3) compute acoustic features, and (4) estimate sound pressure level. This workflow opens the possibility to scan large audio datasets and use powerful machine learning techniques, allowing to measure acoustic properties and identify key patterns in all kinds of soundscapes.

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
   
Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`


Citing scikit-maad
~~~~~~~~~~~~~~~~~~

If you find this package useful in your research, we would appreciate citations to:

.. [#] Ulloa, J. S., Haupert, S., Latorre, J. F., Aubin, T., & Sueur, J. (2021). scikit-maad: An open-source and modular toolbox for quantitative soundscape analysis in Python. Methods in Ecology and Evolution, 2041-210X.13711. https://doi.org/10.1111/2041-210X.13711

About the authors
~~~~~~~~~~~~~~~~~

This work started in 2016 at the Museum National d'Histoire Naturelle (MNHN) 
in Paris, France. It was initiated by **Juan Sebastian Ulloa**, supervised by 
**Jérôme Sueur** and **Thierry Aubin** at the Muséum National d'Histoire Naturelle 
and the Université Paris-Saclay respectively. Python functions have been added by 
**Sylvain Haupert**, **Juan Felipe Latorre**, and **Juan Sebastian Ulloa**. 
