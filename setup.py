#!/usr/bin/env python
#
# Copyright (C) 2018 Juan Sebastian ULLOA <lisofomia@gmail.com>
#                    Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
# License: BSD 3 clause

import os
import sys
import subprocess

from numpy.distutils.core import setup

DISTNAME            = 'scikit-maad'
DESCRIPTION         = 'Set of functions to extract features in a non-supervised manner from an audio recording',
LONG_DESCRIPTION    = open('README.md').read()
MAINTAINER          = 'Juan Sebastian ULLOA and Sylvain HAUPERT',
MAINTAINER_EMAIL    = 'lisofomia@gmail.com and sylvain.haupert@mnhn.fr',
URL                 = 'https://github.com/'
LICENSE             = 'BSD 3 Clause'
DOWNLOAD_URL        = URL
PACKAGE_NAME        = 'maad'
EXTRA_INFO          = dict(
    install_requires=['numpy', 'scipy','scikit-image','scikit-learn','pandas'],
    classifiers=['Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD 3 clause',
                 'Topic :: Scientific/Biology'
                 'Programming Language :: Python :: 3',
                 'Operating System :: OS Independent']
)


def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg: "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage(PACKAGE_NAME)
    return config

def get_version():
    """Obtain the version number"""
    import imp
    mod = imp.load_source('version', os.path.join(PACKAGE_NAME, 'version.py'))
    return mod.__version__

# Documentation building command
try:
    from sphinx.setup_command import BuildDoc as SphinxBuildDoc
    class BuildDoc(SphinxBuildDoc):
        """Run in-place build before Sphinx doc build"""
        def run(self):
            ret = subprocess.call([sys.executable, sys.argv[0], 'build_ext', '-i'])
            if ret != 0:
                raise RuntimeError("Building Scipy failed!")
            SphinxBuildDoc.run(self)
    cmdclass = {'build_sphinx': BuildDoc}
except ImportError:
    cmdclass = {}

# Call the setup function
if __name__ == "__main__":
    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          include_package_data=True,
          test_suite="nose.collector",
          cmdclass=cmdclass,
          version=get_version(),
**EXTRA_INFO)   
    
    
    
    