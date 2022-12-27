Installation instructions
-------------------------

**scikit-maad** is built on top of popular Python packages for scientific computing and digital signal processing, such as numpy, scipy, pandas, scikit-image and resampy. Package managers like pip and conda can take care of these automatically.


1. Standard installation
~~~~~~~~~~~~~~~~~~~~~~~~

**scikit-maad** is available at the Python software repository `PyPI <https://pypi.org/>`_. The simplest way to install it is using **pip**, the standard installer for Python. To install, run the following command in your Python environment::

    pip install scikit-maad

If you use the Conda package manager you may need to first install pip and then run the installation::
    
    conda install pip
    pip install scikit-maad

2. Development installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can install the development version if you want to work on the latest version or develop new features. First uninstall any existing installations::

    pip uninstall scikit-maad

To install the latest deveopment version from source clone the master repository from `GitHub <https://github.com/scikit-maad/scikit-maad>`_ ::

    git clone https://github.com/scikit-maad/scikit-maad.git

Then, from the top-level folder call::

    python setup.py install
    
To update the installation pull the latest modifications from the repository and reinstall::
    
    git pull
    python setup.py install
    
If you do not have git installed, download the latest ZIP version from `GitHub <https://github.com/scikit-maad/scikit-maad>`_, in the tab named **Code**, look for the link **Download ZIP**. In the download location call::
    
    pip install scikit-maad-production.zip
    

