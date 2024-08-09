
# scikit-maad

<div align="center">
    <img src="https://raw.githubusercontent.com/scikit-maad/scikit-maad/production/docs/logo/maad_key_visual_blue.png" alt="scikit-maad logo"/>
</div>

**scikit-maad** is an open source Python package dedicated to the quantitative analysis of environmental audio recordings. This package was designed to 
1. load and process digital audio, 
2. segment and find regions of interest, 
3. compute acoustic features, and 
4. estimate sound pressure level. 

This workflow opens the possibility to scan large audio datasets and use powerful machine learning techniques, allowing to measure acoustic properties and identify key patterns in all kinds of soundscapes.

[![PyPI version](https://badge.fury.io/py/scikit-maad.svg)](https://badge.fury.io/py/scikit-maad)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/148142520.svg)](https://zenodo.org/badge/latestdoi/148142520)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Downloads](https://static.pepy.tech/badge/scikit-maad)](https://pepy.tech/project/scikit-maad)
[![Citation Badge](https://api.juleskreuer.eu/citation-badge.php?doi=10.1111/2041-210X.13711)](https://juleskreuer.eu/projekte/citation-badge/)
<!--[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)-->

## Operating Systems

`scikit-maad` seamlessly supports Linux, macOS, and Windows operating systems.

## Interpreter

`scikit-maad` requires one of these interpreters:

- Python >= 3.8 < 3.11

## Packages dependency

`scikit-maad` requires these Python packages to run:

- matplotlib >=3.6
- numpy >= 1.21
- pandas >= 1.5
- resampy >= 0.4
- scikit-image >= 0.19
- scipy >= 1.8

## Installing from PyPI

`scikit-maad` is hosted on PyPI. The easiest way to install the package is using `pip` the standard package installer for Python:

```bash
$ pip install scikit-maad
```

## Quick start

The package is imported as `maad`. To use scikit-maad tools, audio must be loaded as a numpy array. The function `maad.sound.load` is a simple and effective way to load audio from disk. For example, download the [spinetail audio example](https://raw.githubusercontent.com/scikit-maad/scikit-maad/production/data/spinetail.wav) to your working directory. You can load it and then apply any analysis to find regions of interest or characterize your audio signals:

```python
from maad import sound, rois
s, fs = sound.load('spinetail.wav')
rois.find_rois_cwt(s, fs, flims=(4500,8000), tlen=2, th=0, display=True)
```
## For advance users
### Installing from source

If you are interested in developing new features for `scikit-maad` or working with the latest version, clone and install it:

```bash
$ git clone https://github.com/scikit-maad/scikit-maad.git
$ cd scikit-maad
$ pip install --editable .
```

### Running tests

Install the test requirements:

```bash
$ pip install pytest
```

And run the tests:

```bash
$ cd scikit-maad
$ pytest
```

## Examples and documentation
- See https://scikit-maad.github.io for a complete reference manual and example gallery.

Runnin all examples requires to install the following packages :
- `scikit-learn`, a popular Python package for machine learning: [link](https://scikit-learn.org/stable/install.html)
- `librosa`, a popular package for audio and music analysis: [link](https://librosa.org/doc/latest/install.html)
- `tqdm`, a package that provides a fast, extensible progress bar for loops and other iterable tasks: [link](https://pypi.org/project/tqdm/)
  
- In depth information related to the Multiresolution Analysis of Acoustic Diversity implemented in scikit-maad was published in: Ulloa, J. S., Aubin, T., Llusia, D., Bouveyron, C., & Sueur, J. (2018). [Estimating animal acoustic diversity in tropical environments using unsupervised multiresolution analysis](https://doi.org/10.1016/j.ecolind.2018.03.026). Ecological Indicators, 90, 346–355

## Citing this work

If you find `scikit-maad` usefull for your research, please consider citing it as:

- Ulloa, J. S., Haupert, S., Latorre, J. F., Aubin, T., & Sueur, J. (2021). scikit‐maad: An open‐source and modular toolbox for quantitative soundscape analysis in Python. Methods in Ecology and Evolution, 2041-210X.13711. https://doi.org/10.1111/2041-210X.13711

or use our [citing file](https://raw.githubusercontent.com/scikit-maad/scikit-maad/production/CITATION.bib) for custom citation formats.

## Feedback and contributions
Improvements and new features are greatly appreciated. If you would like to contribute submitting issues, developing new features or making improvements to `scikit-maad`, please refer to our [contributors guide](https://raw.githubusercontent.com/scikit-maad/scikit-maad/production/CONTRIBUTING.md). 
To create a positive social atmosphere for our community, we ask contributors to adopt and enforce our [code of conduct](https://raw.githubusercontent.com/scikit-maad/scikit-maad/production/CODE_OF_CONDUCT.md).

## About the project
In 2018, we began to translate a set of audio processing functions from Matlab to an open-source programming language, namely, Python. These functions provided the necessary tools to replicate the Multiresolution Analysis of Acoustic Diversity (MAAD), a method to estimate animal acoustic diversity using unsupervised learning (Ulloa et al., 2018). We soon realized that Python provided a suitable environment to extend these core functions and to develop a flexible toolbox for our research. During the past few years, we added over 50 acoustic indices, plus a module to estimate the sound pressure level of audio events. Furthermore, we updated, organized, and fully documented the code to make this development accessible to a much wider audience. This work was initiated by [Juan Sebastian Ulloa](https://www.researchgate.net/profile/Juan_Ulloa), supervised by Jérôme Sueur and Thierry Aubin at the [Muséum National d'Histoire Naturelle](http://isyeb.mnhn.fr/fr) and the [Université Paris Saclay](http://neuro-psi.cnrs.fr/) respectively. Python functions have been added by [Sylvain Haupert](https://www.researchgate.net/profile/Sylvain_Haupert), [Juan Felipe Latorre](https://www.researchgate.net/profile/Juan_Latorre_Gil) ([Universidad Nacional de Colombia](https://unal.edu.co/)) and Juan Sebastián Ulloa ([Instituto de Investigación de Recursos Biológicos Alexander von Humboldt](http://www.humboldt.org.co/)). For an updated list of collaborators, check the [contributors list](https://github.com/scikit-maad/scikit-maad/graphs/contributors).

## License
To support reproducible research, the package is released under the [BSD open-source licence](https://raw.githubusercontent.com/scikit-maad/scikit-maad/production/LICENSE.md), which allows unrestricted redistribution for commercial and private use.
