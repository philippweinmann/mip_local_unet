#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='mip_unet',
      version='0.1',
      description='My model for the MIP project',
      author='Philipp Weinmann',
      author_email='philipp.weinmann71@gmail.com',
      packages=find_packages(),
      install_requires=[
        # Dependencies go here
        'numpy', 
        "matplotlib==3.9.2", 
        "scikit-learn==1.5.2", 
        "scikit-image", 
        "torch==2.3.1", 
        "opencv_python_headless==4.10.0.84", 
        "scipy==1.14.1",
        "nibabel",
        "optuna",
        "torch",
    ],
     )