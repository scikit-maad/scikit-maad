#!/usr/bin/env python
# Copyright (C) 2018 Juan Sebastian ULLOA <jseb.ulloa@gmail.com>
#                    Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
# License: BSD 3 clause

import os
import textwrap
from setuptools import setup, find_packages, Command
from importlib.machinery import SourceFileLoader

version = SourceFileLoader(
    'maad.version',
    'maad/version.py'
).load_module()

with open('README.md', 'r', encoding="utf8") as read_me_file:
    long_description = read_me_file.read()


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
    name='scikit-maad',
    version=version.__version__,  # Specified at maad/version.py file
    packages=find_packages(),
    author='Juan Sebastian Ulloa and Sylvain Haupert',
    author_email='jseb.ulloa@gmail.com, sylvain.haupert@mnhn.fr',
    maintainer='Juan Sebastian Ulloa and Sylvain Haupert',
    description='scikit-maad, soundscape analysis in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='BSD 3 Clause',
    keywords=[
        'ecoacoustics',
        'bioacoustics',
        'ecology',
        'sound pressure level',
        'signal processing',
    ],
    url='https://github.com/scikit-maad/scikit-maad',
    project_urls={'Documentation': 'https://scikit-maad.github.io'},
    platforms='OS Independent',
    cmdclass={'clean': CleanCommand},
    license_file='LICENSE',
    python_requires='>=3.5',
    install_requires=[
        'numpy>=1.19',
        'scipy>=1.5',
        'scikit-image>=0.17',
        'pandas>=1.1',
        'resampy>=0.2',
        'matplotlib'
    ],
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
