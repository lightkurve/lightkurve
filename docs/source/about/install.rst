.. _installation:

*********************
Installing Lightkurve
*********************

Using conda
===========

The easiest way to install *Lightkurve* and all of its dependencies is to use
the ``conda`` package manager, which is part of the 
`Anaconda Python <https://www.continuum.io/downloads>`_ distribution.
With ``conda`` installed, simply run the following command in a terminal window::

    $ conda install --channel conda-forge lightkurve

If you have a previous version of *Lightkurve* installed,
you can update it using::

    $ conda update lightkurve

To verify which version of *Lightkurve* you have installed, run::

    $ python -c "import lightkurve; print(lightkurve.__version__)"


Using pip
=========

An alternative way to install *Lightkurve* is to use the ``pip`` package
manager, which is a standard part of all Python distributions.
To install *Lightkurve*, run the following command in a terminal window::

    $ pip install lightkurve --upgrade

The ``--upgrade`` flag is optional, but recommended if you already
have *Lightkurve* installed and want to upgrade to the latest version.

By default, ``pip`` won't install *Lightkurve*'s optional dependencies.
We configured it this way because ``pip`` requires a C compiler to install
some of these optional dependencies, which is not usually available
on Windows systems. If you want to try to install these, you can use::

     $ pip install lightkurve[all]

If you encounter any compilation errors using this command, then we recommend
that you use the ``conda`` package manager instead.


.. note::

    If you encounter a ``PermissionError`` this means that you do not have the
    required administrative access to install new packages to your Python
    installation.  In this case you may consider using the ``--user`` option
    to install the package into your home directory.  You can read more
    about how to do this in the `pip documentation
    <http://www.pip-installer.org/en/1.2.1/other-tools.html#using-pip-with-the-user-scheme>`_.



Requirements
============

*Lightkurve* has the following minimum requirements:

- Python: 2.7, 3.5, 3.6, 3.7, or later.
- Astropy: 1.3 or later.
- Numpy: 1.11 or later.
- Scipy: 0.19 or later.
- Matplotlib: 1.5.3 or later.
- Astroquery 0.3.9 or later.
- Pandas.

A few extra features (interact widgets, PLD correction, and BLS periodograms) require optional dependencies which are not installed by default using ``pip`` (though they are installed if you use ``conda``):

- Astropy: 3.1 or later (for BLS periodograms).
- Bokeh: 1.0 or later (for interactive widgets).
- Scikit-learn and Celerite (for PLD systematics correction).

We recommend using the `Anaconda Python <https://www.continuum.io/downloads>`_
distribution, which will install Python alongside its most common scientific
packages, including all those listed above.
If you install *Lightkurve* using ``conda`` or ``pip`` as explained above, any missing dependencies will be installed automatically.



Installing the development version
==================================

If you want to experiment with the latest development version of
*Lightkurve*, you can install it straight from the master branch on GitHub::

    $ git clone https://github.com/KeplerGO/lightkurve.git
    $ cd lightkurve
    $ pip install -e .

This is recommended for anyone who wants to edit the source code.
Please see our guide on :ref:`contributing to lightkurve<contributing>`
for additional instructions.
