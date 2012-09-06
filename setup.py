# -*- coding: utf-8 -*-

from distutils.core import setup
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
        packages=['spykeutils', 'spykeutils.plot', 'spykeutils.plugin'],
        requires=['scipy', 'quantities', 'neo'],
        author='Robert Pröpper',
        maintainer='Robert Pröpper',
        description='spykeutils: Utilities for analyzing electrophysiological data',
        long_description=open('README.rst', 'r').read(),
        license='MIT License',
        url='https://github.com/rproepp/spykeutils',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics'])