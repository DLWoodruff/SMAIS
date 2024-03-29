# This software is distributed under the 3-clause BSD License.
#!/bin/usr/env python3
import glob
import sys
import os

if sys.version[0] == '2':
    print("Error: This package must be installed with python3")
    sys.exit(1)

from setuptools import find_packages
from distutils.core import setup

packages = find_packages()

# intentionally leaving out mpi4py to help readthedocs
setup(
    name='smais',
    version='0.1.dev0',
    description="smais",
    url='https://github.com/DLWoodruff/SMAIS',
    author='David Woodruff',
    author_email='dlwoodruff@ucdavis.edu',
    packages=packages,
    install_requires=[
        'numpy',
        'scipy',
        'pyomo>=6.7',
        'mpi-sppy',
        'boot-sp'
    ]
)
