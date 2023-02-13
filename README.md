<div align="center">
    <img src="./docs/logo/maad_key_visual_blue.png" alt="drawing"/>
</div>

**scikit-maad** is an open-source Python package for the quantitative analysis
of environmental audio recordings. This package was designed to:

1. load and process digital audio,
2. segment and find regions of interest,
3. compute acoustic features, and
4. estimate sound pressure level.

This workflow opens the possibility to scan large audio datasets and use
powerful machine learning techniques, allowing to measure acoustic properties
and identify key patterns in all kinds of soundscapes.

[![DOI](https://zenodo.org/badge/148142520.svg)](https://zenodo.org/badge/latestdoi/148142520)
[![Downloads](https://static.pepy.tech/badge/scikit-maad)](https://pepy.tech/project/scikit-maad)

## Operating Systems

**scikit-maad** supports these operating systems:

- Linux (tested on Ubuntu in CI)
- macOS (Intel CPUs only; Apple Silicon not supported)
- Windows

## Interpreter

**scikit-maad** requires one of these interpreters:

- Python >= 3.5 < 3.11

## Cloning

**scikit-maad** is hosted on GitHub and can be cloned like so:

```bash
git clone https://github.com/scikit-maad/scikit-maad.git
cd scikit-maad
```

> Note: We expect all following commands to be run in the `scikit-maad` folder.

## Virtual Environment

**scikit-maad** should be installed into a virtual environment created like so:

```bash
python3.10 -m venv venv
venv/bin/python -m pip install --upgrade pip setuptools wheel
```

## Packages

**scikit-maad** requires these Python packages to run:

- matplotlib
- NumPy >= 1.19
- pandas>=1.1
- resampy>=0.2
- scikit-image >= 0.17
- SciPy >= 1.5
- scikit-learn

## Installing from PyPI

If you want to use **scikit-maad** in your project, install it from PyPI:

```bash
venv/bin/python -m pip install scikit-maad
```

## Installing from source

If you want to work on **scikit-maad** itself, clone and install it:

```bash
venv/bin/python -m pip install --editable .
```

## Running tests

Install the test requirements:

```bash
venv/bin/python -m pip install pytest-cov
```

Actually run the tests:

```bash
venv/bin/python -m pytest --cov tests --cov maad --cov example_gallery
```

## Examples and documentation

- See https://scikit-maad.github.io for a complete reference manual and example
  gallery.
- In depth information related to the Multi-resolution Analysis of Acoustic
  Diversity implemented in scikit-maad was published in: Ulloa, J. S., Aubin,
  T., Llusia, D., Bouveyron, C., & Sueur, J. (2018).
  [Estimating animal acoustic diversity in tropical environments using unsupervised multiresolution analysis](https://doi.org/10.1016/j.ecolind.2018.03.026).
  Ecological Indicators, 90, 346–355

## Citing this work

If you find **scikit-maad** useful for your research, please consider citing it as:

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
```

## Contributions and bug reports

Improvements and new features are greatly appreciated. If you would like to
contribute developing new features or making improvements to the available
package, please refer to our
[wiki](https://github.com/scikit-maad/scikit-maad/wiki/How-to-contribute-to-scikit-maad).
Bug reports and especially tested patches may be submitted directly to the
[bug tracker](https://github.com/scikit-maad/scikit-maad/issues).

## About the project

In 2018, we began to translate a set of audio processing functions from Matlab
to an open-source programming language, namely, Python. These functions
provided the necessary tools to replicate the Multiresolution Analysis of
Acoustic Diversity (MAAD), a method to estimate animal acoustic diversity using
unsupervised learning (Ulloa et al., 2018). We soon realized that Python
provided a suitable environment to extend these core functions and to develop a
flexible toolbox for our research. During the past few years, we added over 50
acoustic indices, plus a module to estimate the sound pressure level of audio
events. Furthermore, we updated, organized, and fully documented the code to
make this development accessible to a much wider audience. This work was
initiated by
[Juan Sebastian Ulloa](https://www.researchgate.net/profile/Juan_Ulloa),
supervised by Jérôme Sueur and Thierry Aubin at the
[Muséum National d'Histoire Naturelle](http://isyeb.mnhn.fr/fr) and the
[Université Paris Saclay](http://neuro-psi.cnrs.fr/) respectively. Python
functions have been added by
[Sylvain Haupert](https://www.researchgate.net/profile/Sylvain_Haupert),
[Juan Felipe Latorre](https://www.researchgate.net/profile/Juan_Latorre_Gil)
([Universidad Nacional de Colombia](https://unal.edu.co/)) and Juan Sebastián
Ulloa
([Instituto de Investigación de Recursos Biológicos Alexander von Humboldt](http://www.humboldt.org.co/)).
