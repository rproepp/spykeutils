Requirements
============
Spykeutils is a pure Python package and therefore easy to install. It depends
on the following additional packages:

* Python_ >= 2.7
* neo_ >= 0.2.1
* scipy_
* guiqwt_ >= 2.1.4 (Optional, for plotting)
* tables_ (Optional, for analysis results data management. Also known as
  PyTables.)
* scikit-learn_ (Optional, for spike sorting quality analysis using Gaussian
  cluster overlap.)

Please see the respective websites for instructions on how to install them if
they are not present on your computer. If you use Linux, you might not
have access rights to your Python package installation directory, depending
on your configuration. In this case, you will have to execute all shell
commands in this section with administrator privileges, e.g. by using
``sudo``.

Download and Installation
=========================
The easiest way to get spykeutils is from the Python Package Index.
If you have pip_ installed::

$ pip install spykeutils

Alternatively, if you have setuptools_::

$ easy_install spykeutils

Users of NeuroDebian_ or its repositories (available for Debian and Ubuntu)
can also install spykeutils using the package manager instead of pip_. The
package is also available directly in recent Debian and Ubuntu installations,
but might not be the most recent version. Install with::

$ sudo apt-get install python-spykeutils

Alternatively, you can get the latest version directly from GitHub at
https://github.com/rproepp/spykeutils.

The master branch always contains the current stable version. If you want the
latest development version, use the develop branch (selected by default).
You can download the repository from the GitHub page
or clone it using git and then install from the resulting folder::

$ python setup.py install

Usage
=====
For the most part, spykeutils is a collection of functions that work on
Neo objects. Many functions also take quantities as parameters. Therefore,
make sure to get an overview of :mod:`neo` and :mod:`quantities` before using
spykeutils. Once you are familiar with these packages, have a look at the
:ref:`examples` or head to the :ref:`apiref` to browse the contents of
spykeutils.

.. _`Python`: http://python.org
.. _`neo`: http://neo.readthedocs.org
.. _`guiqwt`: http://packages.python.org/guiqwt
.. _`tables`: http://www.pytables.org
.. _`quantities`: http://pypi.python.org/pypi/quantities
.. _`scikit-learn`: http://scikit-learn.org
.. _`pip`: http://pypi.python.org/pypi/pip
.. _`scipy`: http://scipy.org
.. _`setuptools`: http://pypi.python.org/pypi/setuptools
.. _`NeuroDebian`: http://neuro.debian.net
