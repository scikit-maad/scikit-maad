.. scikit-maad documentation master file, created by
   sphinx-quickstart on Tue Aug 18 11:29:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to scikit-maad's documentation!
=======================================

**scikit-maad** is a free, open-source and modular toolbox to **analyze 
ecoacoustics datasets** in Python 3. This package was designed to bring 
flexibility to (1) **find regions of interest**, and (2) to compute **acoustic 
features** in audio recordings. This workflow opens the possibility to use 
powerfull **machine learning** algorithms through **scikit-learn**, 
allowing to identify key patterns in all kind of soundscapes.


About the authors
~~~~~~~~~~~~~~~~~

This work started in 2016 at the Museum National d'Histoire Naturelle (MNHN) 
in Paris, France. It was initiated by **Juan Sebastian Ulloa**, supervised by 
**Jérôme Sueur** and **Thierry Aubin** at the Muséum National d'Histoire Naturelle 
and the Université Paris Saclay respectively. Python functions were added by 
**Sylvain Haupert** and **Chloe Huetz** in 2018. New features are currently being 
developped and a stable release will be available by the end of 2019.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   install
   tutorial
   
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :caption: Documentation:
   :recursive:
   
   cluster
   ecoacoustics
   features
   rois
   sound
   util
   
   
   
.. toctree::
   :maxdepth: 2
   :caption: Example Gallery:
   
   Example


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
