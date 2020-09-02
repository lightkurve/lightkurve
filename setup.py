#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("python setup.py bdist_wheel")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/lightkurve*")
    sys.exit()

# Load the __version__ variable without importing the package already
exec(open('lightkurve/version.py').read())

# DEPENDENCIES
# 1. What are the required dependencies?
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
# 2. What dependencies required to run the unit tests? (i.e. `pytest --remote-data`)
with open('requirements-test.txt') as f:
    tests_require = f.read().splitlines()
extras_require = {"test": tests_require}

setup(name='lightkurve',
      version=__version__,
      description="A friendly package for Kepler & TESS time series analysis "
                  "in Python.",
      long_description=open('README.rst').read(),
      author='Geert Barentsen',
      author_email='geert@barentsen.be',
      url='https://docs.lightkurve.org',
      license='MIT',
      package_dir={
            'lightkurve': 'lightkurve',
            'lightkurve.io': 'lightkurve/io',
            'lightkurve.correctors': 'lightkurve/correctors',
            'lightkurve.seismology': 'lightkurve/seismology',
            'lightkurve.prf': 'lightkurve/prf'},
      packages=['lightkurve', 'lightkurve.io', 'lightkurve.correctors',
                'lightkurve.seismology', 'lightkurve.prf'],
      install_requires=install_requires,
      extras_require=extras_require,
      setup_requires=['pytest-runner'],
      tests_require=tests_require,
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
