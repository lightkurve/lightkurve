#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/lightkurve*")
    sys.exit()


# Load the __version__ variable without importing the package already
exec(open('lightkurve/version.py').read())

setup(name='lightkurve',
      version=__version__,
      description="A beautiful package for astronomical flux time series"
                  "analysis in Python.",
      long_description=open('README.rst').read(),
      author='KeplerGO',
      author_email='keplergo@mail.arc.nasa.gov',
      license='MIT',
      packages=['lightkurve'],
      install_requires=['numpy>=1.11', 'astropy>=1.3', 'scipy>=0.17.0',
                        'matplotlib>=1.5.3', 'tqdm', 'oktopus', 'bs4',
                        'requests', 'astroquery>=0.3.7'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'pytest-cov', 'pytest-remotedata'],
      include_package_data=True,
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
    )
