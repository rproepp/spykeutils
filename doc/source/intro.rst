Requirements
============
Spykeutils is a pure Python package and therefore easy to install. It depends
on the following additional packages:

* Python_ >= 2.7
* neo_ >= 0.2.1
* scipy_
* guiqwt_ (Optional, for plotting)
* tables_ (Optional, for analysis results data management. Also known as
  PyTables.)

Please see the respective websites for instructions on how to install them if
they are not present on your computer. If you use Linux, you might not
have access rights to your Python package installation directory, depending
on your configuration. In this case, you will have to execute all shell
commands in this section with administrator privileges, e.g. by using
``sudo``.

.. note::
    If you have installed neo_ before, please make sure you have the most
    recent version of the package, e.g. by running::

    $ pip install -U neo

    or::

    $ easy_install -U neo

Download and Installation
=========================
The easiest way to get spykeutils is from the Python Package Index.
If you have pip_ installed::

$ pip install spykeutils

Alternatively, if you have setuptools_::

$ easy_install spykeutils

Alternatively, you can get the latest version directly from GitHub at
https://github.com/rproepp/spykeutils.

The master branch (selected by default) always contains the current stable
version. If you want the latest development version (not recommended unless
you need some features that do not exist in the stable version yet), select
the develop branch. You can download the repository from the GitHub page
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

.. _`Python`: http://python.org/
.. _`neo`: http://neo.readthedocs.org/
.. _`guiqwt`: http://packages.python.org/guiqwt/
.. _`tables`: http://www.pytables.org/
.. _`quantities`: http://pypi.python.org/pypi/quantities
.. _`pip`: http://pypi.python.org/pypi/pip
.. _`scipy`: http://scipy.org/
.. _`setuptools`: http://pypi.python.org/pypi/setuptools