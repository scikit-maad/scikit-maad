Installation instructions
-------------------------

**scikit-maad** is built on top of popular Python packages for scientific computing and digital signal processing, such as numpy, scipy, pandas, scikit-image and resampy. Package managers like pip and conda can take care of these automatically.


1. Standard installation
~~~~~~~~~~~~~~~~~~~~~~~~

**scikit-maad** is available at the Python software repository `PyPI <https://pypi.org/>`_. The simplest way to install it is using **pip**, the standard installer for Python. To install, run the following command in your Python environment::

    pip install scikit-maad

2. Development installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install the latest deveopment version from source clone the master repository from `GitHub <https://github.com/scikit-maad/scikit-maad>`_ ::

    git clone https://github.com/scikit-maad/scikit-maad.git
    cd scikit-maad  # enter the cloned directory

Install the listed dependencies with::
    
    pip install -r requirements.txt

Then, from the top-level folder call::

    pip install -e .
    
