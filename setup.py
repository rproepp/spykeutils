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
        long_description=open('README.rst', 'r').read(),
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