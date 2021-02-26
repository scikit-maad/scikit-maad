<img src="logo_maad.png" alt="drawing" width="500"/>

**scikit-maad** is a free, open-source and modular toolbox to **analyze ecoacoustics datasets** in Python 3. This package was designed to bring flexibility to (1) **find regions of interest**, and (2) to compute **acoustic features** in audio recordings. This workflow opens the possibility to use powerfull **machine learning** algorithms through **scikit-learn**, allowing to identify key patterns in all kind of soundscapes.

[![DOI](https://zenodo.org/badge/148142520.svg)](https://zenodo.org/badge/latestdoi/148142520)

## Installation
scikit-maad dependencies:

- Python >= 3.5
- NumPy >= 1.13
- SciPy >= 0.18
- scikit-image >= 0.14

**scikit-maad** is hosted on PyPI. To install, run the following command in your Python environment:

```bash
$ pip install scikit-maad
```

To install the latest version from source clone the master repository and from the top-level folder call:

```bash
$ python setup.py install
```

## Examples and documentation
- See https://scikit-maad.github.io/scikit-maad for a complete reference manual and example gallery.
- In depth information related to the Multiresolution Analysis of Acoustic Diversity implemented in scikit-maad was published in: Ulloa, J. S., Aubin, T., Llusia, D., Bouveyron, C., & Sueur, J. (2018). [Estimating animal acoustic diversity in tropical environments using unsupervised multiresolution analysis](https://doi.org/10.1016/j.ecolind.2018.03.026). Ecological Indicators, 90, 346–355

## Contributions and bug report
Improvements and new features are greatly appreciated. If you would like to contribute developing new features or making improvements to the available package, please refer to our [wiki](https://github.com/scikit-maad/scikit-maad/wiki/How-to-contribute-to-scikit-maad). Bug reports and especially tested patches may be submitted directly to the [bug tracker](https://github.com/scikit-maad/scikit-maad/issues). 

## About the authors
This work started in 2016 at the Museum National d'Histoire Naturelle (MNHN) in Paris, France. It was initiated by [Juan Sebastian Ulloa](https://www.researchgate.net/profile/Juan_Ulloa), supervised by Jérôme Sueur and Thierry Aubin at the [Muséum National d'Histoire Naturelle](http://isyeb.mnhn.fr/fr) and the [Université Paris Saclay](http://neuro-psi.cnrs.fr/) respectively. Python functions were added by [Sylvain Haupert](https://www.researchgate.net/profile/Sylvain_Haupert), [Juan Felipe Latorre](https://www.researchgate.net/profile/Juan_Latorre_Gil) and Juan Sebastián Ulloa in 2018. New features are currently being developped and a stable release will be available by 2021.
