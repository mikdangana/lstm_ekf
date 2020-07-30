#!/usr/bin/env python3

#from distutils.core import setup
from setuptools import setup

setup(name='Hyscale8',
      version='1.0',
      description='Hyscale Kubernetes Plugin',
      author='MSRG',
      author_email='michael.dangana@mail.utoronto.ca',
      url='https://github.com/msrg/hyscale/',
      packages=['hyscale'],
      install_requires=[
          'scipy'
          'matplotlib',
          'numpy',
          'Cython'
      ]
     )
