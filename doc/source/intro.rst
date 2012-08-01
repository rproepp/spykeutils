Dependencies
============
spykeutils is based on the :mod:`neo` framework, so it depends on the neo
package. This means it also carries neo's dependencies -
`scipy <http://scipy.org>`_ and :mod:`quantities`.
In addition, Python 2.7 or newer is required.

For the :mod:`plot` subpackage, :mod:`guiqwt` is required. The main
:mod:`spykeutils` can be used without guiqwt, but plotting will be
unavailable. However, many spykeutils functions return data that can
easily be plotted with an arbitrary plotting library, e.g.
`matplotlib <http://matplotlib.sourceforge.net>`_.

Installation
============
At the moment, spykeutils is not supposed to be installed. Just put it in
your python path and you're done. A setup will be added in the near future.

Currently, the neo version in pypi contains bugs that affect some of the
functions in spykeutils. Please install the most recent version from
`GitHub <https://github.com/python-neo/python-neo>`_.

Usage
=====
For the most part, spykeutils consists of independent functions that work on
neo objects. Many functions also take quantities as parameters. Therefore,
make sure to have an overview :mod:`neo` and :mod:`quantities` before using
spykeutils. Once you are familiar with these packages, head to the
:ref:`apiref` to browse the contents of spykeutils.