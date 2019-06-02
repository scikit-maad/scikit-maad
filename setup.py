#!/usr/bin/env python
#
# Copyright (C) 2018 Juan Sebastian ULLOA <jseb.ulloa@gmail.com>
#                    Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
# License: BSD 3 clause

import os
import textwrap
from setuptools import setup, find_packages, Command


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
      version = '0.1.4',
      #packages = find_namespace_packages(include=['maad.*']),
      packages = find_packages(),
      author = 'Juan Sebastian ULLOA and Sylvain HAUPERT',
      author_email = 'jseb.ulloa@gmail.com, sylvain.haupert@mnhn.fr',
      maintainer = 'Juan Sebastian ULLOA and Sylvain HAUPERT',
      description = 'scikit-maad is a modular toolbox to analyze ecoacoustics datasets',
      long_description = 'scikit-maad is a modular toolbox to analyze ecoacoustics datasets in Python 3. This package was designed to bring flexibility to find regions of interest, and to compute acoustic features in audio recordings. This workflow opens the possibility to use powerfull machine learning algorithms through scikit-learn, allowing to identify key patterns in all kind of soundscapes.',
      license = 'BSD 3 Clause',
      keywords = ['ecoacoustics', 'machine learning', 'ecology', 'wavelets', 'signal processing'],
      url = 'https://github.com/scikit-maad/scikit-maad',
      platform = 'OS Independent',
      cmdclass={'clean': CleanCommand},
      license_file = 'LICENSE',                     

      install_requires = ['docutils>=0.3', 'numpy>=1.13', 'scipy>=0.18', 
                          'scikit-image>=0.14', 'scikit-learn>=0.18',
                          'pandas>=0.23.4'],

      classifiers=textwrap.dedent("""
        Development Status :: 4 - Beta
        Intended Audience :: Science/Research
        License :: OSI Approved :: BSD License
        Operating System :: OS Independent
        Programming Language :: Python :: 3.5
        Programming Language :: Python :: 3.6
        Programming Language :: Python :: 3.7
        Topic :: Scientific/Engineering :: Artificial Intelligence 
        """).strip().splitlines()
       )
