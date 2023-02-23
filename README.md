
# scikit-maad

<div align="center">
    <img src="./docs/logo/maad_key_visual_blue.png" alt="drawing"/>
</div>

**scikit-maad** is an open source Python package dedicated to the quantitative analysis of environmental audio recordings. This package was designed to (1) load and process digital audio, (2) segment and find regions of interest, (3) compute acoustic features, and (4) estimate sound pressure level. This workflow opens the possibility to scan large audio datasets and use powerful machine learning techniques, allowing to measure acoustic properties and identify key patterns in all kinds of soundscapes.

[![DOI](https://zenodo.org/badge/148142520.svg)](https://zenodo.org/badge/latestdoi/148142520)
[![Downloads](https://static.pepy.tech/badge/scikit-maad)](https://pepy.tech/project/scikit-maad)
[![PyPI version](https://badge.fury.io/py/scikit-maad.svg)](https://badge.fury.io/py/scikit-maad)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Citation Badge](https://api.juleskreuer.eu/citation-badge.php?doi=10.1111/2041-210X.13711)](https://juleskreuer.eu/projekte/citation-badge/)
<!--[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)-->

## Installation

**scikit-maad** is hosted on PyPI. To easiest way to install the package is using `pip` the standard package installer for Python:

```bash
$ pip install scikit-maad
```

## Quick start

The package is imported as `maad`:

```python
from maad import sound, rois
```

To use scikit-maad tools, audio must be loaded as a numpy array. The function `maad.sound.load` is a simple and effective way to load audio from disk. For example, download the [spinetail audio example](https://github.com/scikit-maad/scikit-maad/blob/production/data/spinetail.wav) to your working directory and type:

```python
s, fs = sound.load_url('spinetail')
```

You can then apply any analysis to find regions of interest or characterize your audio signals.

```python
rois.find_rois_cwt(s, fs, flims=(4500,8000), tlen=2, th=0, display=True)
```

## Examples and documentation
- See https://scikit-maad.github.io for a complete reference manual and example gallery.
- In depth information related to the Multiresolution Analysis of Acoustic Diversity implemented in scikit-maad was published in: Ulloa, J. S., Aubin, T., Llusia, D., Bouveyron, C., & Sueur, J. (2018). [Estimating animal acoustic diversity in tropical environments using unsupervised multiresolution analysis](https://doi.org/10.1016/j.ecolind.2018.03.026). Ecological Indicators, 90, 346–355

## Citing this work

If you find scikit-maad usefull for your research, please consider citing it as:

- Ulloa, J. S., Haupert, S., Latorre, J. F., Aubin, T., & Sueur, J. (2021). scikit‐maad: An open‐source and modular toolbox for quantitative soundscape analysis in Python. Methods in Ecology and Evolution, 2041-210X.13711. https://doi.org/10.1111/2041-210X.13711

```bibtex

@article{ulloa_etal_scikitmaad_2021,
	title = {scikit‐maad: {An} open‐source and modular toolbox for quantitative soundscape analysis in {Python}},
	issn = {2041-210X, 2041-210X},
	shorttitle = {scikit‐maad},
	url = {https://onlinelibrary.wiley.com/doi/10.1111/2041-210X.13711},
	doi = {10.1111/2041-210X.13711},
	language = {en},
	urldate = {2021-10-04},
	journal = {Methods in Ecology and Evolution},
	author = {Ulloa, Juan Sebastián and Haupert, Sylvain and Latorre, Juan Felipe and Aubin, Thierry and Sueur, Jérôme},
	month = sep,
	year = {2021},
	pages = {2041--210X.13711},
}
````

## Feedback and contributions
Improvements and new features are greatly appreciated. If you would like to contribute submitting issues, developing new features or making improvements to `scikit-maad`, please refer to our [contributors guide](CONTRIBUTING.md). To create a positive social atmosphere for our community, we ask contributors to adopt and enforce our [code of conduct](CODE_OF_CONDUCT.md).

## About the project
In 2018, we began to translate a set of audio processing functions from Matlab to an open-source programming language, namely, Python. These functions provided the necessary tools to replicate the Multiresolution Analysis of Acoustic Diversity (MAAD), a method to estimate animal acoustic diversity using unsupervised learning (Ulloa et al., 2018). We soon realized that Python provided a suitable environment to extend these core functions and to develop a flexible toolbox for our research. During the past few years, we added over 50 acoustic indices, plus a module to estimate the sound pressure level of audio events. Furthermore, we updated, organized, and fully documented the code to make this development accessible to a much wider audience. This work was initiated by [Juan Sebastian Ulloa](https://www.researchgate.net/profile/Juan_Ulloa), supervised by Jérôme Sueur and Thierry Aubin at the [Muséum National d'Histoire Naturelle](http://isyeb.mnhn.fr/fr) and the [Université Paris Saclay](http://neuro-psi.cnrs.fr/) respectively. Python functions have been added by [Sylvain Haupert](https://www.researchgate.net/profile/Sylvain_Haupert), [Juan Felipe Latorre](https://www.researchgate.net/profile/Juan_Latorre_Gil) ([Universidad Nacional de Colombia](https://unal.edu.co/)) and Juan Sebastián Ulloa ([Instituto de Investigación de Recursos Biológicos Alexander von Humboldt](http://www.humboldt.org.co/)). For an updated list of collaborators, check the [contributors list](https://github.com/scikit-maad/scikit-maad/graphs/contributors).

## License
To support reproducible research, the package is released under the [BSD open-source licence](LICENSE.md), which allows unrestricted redistribution for commercial and private use.
