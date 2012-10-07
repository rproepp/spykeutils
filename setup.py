# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

def find_version():
    try:
        f = open(os.path.join('spykeutils', '__init__.py'), 'r')
        try:
            for line in f:
                if line.startswith('__version__'):
                    rval = line.split()[-1][1:-1]
                    break
        finally:
            f.close()
    except Exception:
        rval = '0'
    return rval

DESC = """Based on the `Neo <http://packages.python.org/neo/>`_ framework,
spykeutils is a Python library for analyzing and plotting data from
neurophysiological recordings. It can be used by itself or in conjunction
with Spyke Viewer, a multi-platform GUI application for navigating
electrophysiological datasets.

For more information, see the documentation at
http://spykeutils.readthedocs.org"""

if __name__ == "__main__":
    setup(
        name="spykeutils",
        version=find_version(),
        packages=find_packages(),
        install_requires=['scipy', 'quantities', 'neo'],
        extras_require = {
            'plot':  ['guiqwt'],
            'plugin': ['tables']
        },
        entry_points = {
            'console_scripts':
                ['spyke-plugin = spykeutils.plugin.start_plugin:main']
        },
        author='Robert Pröpper',
        maintainer='Robert Pröpper',
        description='Utilities for analyzing electrophysiological data',
        long_description=DESC,
        license='BSD',
        url='https://github.com/rproepp/spykeutils',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics'])