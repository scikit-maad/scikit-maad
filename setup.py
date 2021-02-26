#!/usr/bin/env python
#
# Copyright (C) 2018 Juan Sebastian ULLOA <jseb.ulloa@gmail.com>
#                    Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
# License: BSD 3 clause

import os
import textwrap
from setuptools import setup, find_packages, Command
from importlib.machinery import SourceFileLoader

version = SourceFileLoader('maad.version',
                           'maad/version.py').load_module()

with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root.
    Deletes directories ./build, ./dist and ./*.egg-info
    From the terminal type:
        > python setup.py clean
    """
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./*.egg-info')

setup(
      name = 'scikit-maad',
      version = version.__version__,  # Specified at maad/version.py file
      #packages = find_namespace_packages(include=['maad.*']),
      packages = find_packages(),
      author = 'Juan Sebastian Ulloa and Sylvain Haupert',
      author_email = 'jseb.ulloa@gmail.com, sylvain.haupert@mnhn.fr',
      maintainer = 'Juan Sebastian Ulloa and Sylvain Haupert',
      description = 'scikit-maad is a modular toolbox to analyze ecoacoustics datasets',
      long_description = long_description,
      long_description_content_type='text/markdown',
      license = 'BSD 3 Clause',
      keywords = ['ecoacoustics', 'machine learning', 'ecology', 'wavelets', 'signal processing'],
      url = 'https://github.com/scikit-maad/scikit-maad',
      proyect_urls={'Documentation': 'https://scikit-maad.github.io/scikit-maad'},
      platform = 'OS Independent',
      cmdclass={'clean': CleanCommand},
      license_file = 'LICENSE',                     
      python_requires='>=3.6',
      install_requires = ['docutils>=0.3', 
                          'numpy>=1.13', 
                          'scipy>=0.18', 
                          'scikit-image>=0.14', 
                          'scikit-learn>=0.18',
                          'pandas>=0.23.4',
                          'resampy>=0.2'],

      classifiers=textwrap.dedent("""
        Development Status :: 4 - Beta
        Intended Audience :: Science/Research
        License :: OSI Approved :: BSD License
        Operating System :: OS Independent
        Programming Language :: Python :: 3.5
        Programming Language :: Python :: 3.6
        Programming Language :: Python :: 3.7
        Programming Language :: Python :: 3.8
        Topic :: Scientific/Engineering :: Artificial Intelligence 
        """).strip().splitlines()
       )
