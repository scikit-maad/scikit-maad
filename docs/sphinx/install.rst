Installation instructions
-------------------------

**scikit-maad** is a free, open-source and modular Python package to analyze 
ecoacoustics datasets. **scikit-maad** works with other popular Python Scientific Packages. The scikit-maad dependencies are listed below:

- Python >= 3.5
- NumPy >= 1.13
- SciPy >= 0.18
- scikit-image >= 0.14


1. Standard installation
~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to install **scikit-maad** is using **pypi**, the package installer for Python. To install, run the following command in your Python environment::

    pip install scikit-maad

If you use the Conda package manager you may need to first install pip and then run the installation::
    
    conda install pip
    pip install scikit-maad

2. Development installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the development version if you want to work on the latest version or develop new features. First uninstall any existing installations::

    pip uninstall scikit-maad

To install the latest deveopment version from source clone the master repository from `GitHub <https://github.com/scikit-maad/scikit-maad>`_::

    git clone https://github.com/scikit-maad/scikit-maad.git

Then, from the top-level folder call::

    python setup.py install
    
To update the installation pull the latest modifications from the repository and reinstall::
    
    git pull
    python setup.py install
    
If you do not have git installed, download the latest ZIP version from `GitHub <https://github.com/scikit-maad/scikit-maad>`_, in the tab named **Code**, look for the link **Download ZIP**. In the download location call::
    
    pip install scikit-maad-production.zip
    

