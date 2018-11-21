.. _installation:

************
Installation
************

Requirements
============

*Lightkurve* has the following requirements, all of which tend to be
available by default in a modern installation of Python:

- Python: 2.7, 3.5, 3.6 or later.
- Astropy: 1.3 or later.
- Numpy: 1.11 or later.
- Scipy: 0.19 or later.
- Matplotlib: 1.5.3 or later.
- Astroquery 0.3.7 or later.

Optional dependencies:

- Pandas: 0.20 or later.
- Bokeh: 1.0 or later.

We recommend using the `Anaconda Python <https://www.continuum.io/downloads>`_
distribution, which will install Python alongside its most common scientific
packages, including all those listed above.
If you install *lightkurve* using ``pip`` or ``conda`` as explained below, any missing dependencies will be installed automatically.


Installing lightkurve
=====================

Using pip
---------

The easiest way to install or upgrade lightkurve is with ``pip``,
which is standard part of most Python distributions.
To install *lightkurve*, simply run the following command on a terminal window::

    $ pip install lightkurve --upgrade

The ``--upgrade`` flag is optional, but recommended if you already
have *lightkurve* installed and want to upgrade to the latest version.

.. note::

    If you get a ``PermissionError`` this means that you do not have the
    required administrative access to install new packages to your Python
    installation.  In this case you may consider using the ``--user`` option
    to install the package into your home directory.  You can read more
    about how to do this in the `pip documentation
    <http://www.pip-installer.org/en/1.2.1/other-tools.html#using-pip-with-the-user-scheme>`_.


Using conda
-----------

Alternatively, you can use the ``conda`` package manager, which is part of the
`Anaconda Python <https://www.continuum.io/downloads>`_ distribution.
With ``conda`` installed, you can run the following command on a terminal window::

    $ conda install --channel conda-forge lightkurve


Installing the development version
==================================

If you want to experiment with the latest development version of
*lightkurve*, you can install it straight from the master branch on GitHub::

    $ git clone https://github.com/KeplerGO/lightkurve.git
    $ cd lightkurve
    $ pip install -e .

This is recommended for anyone who wants to edit the source code.
Please see our guide on :ref:`contributing to lightkurve<contributing>`
for additional instructions.

Building documentation
======================

.. note::

    Building the documentation is not necessary unless you are
    writing new documentation or do not have internet access, because the
    latest version of the documentation is available online at
    `docs.lightkurve.org <https://docs.lightkurve.org/>`_ .

Building the *lightkurve* documentation requires a few extra packages:

- sphinx
- sphinx-automodapi
- nbsphinx
- `numpydoc <https://github.com/numpy/numpydoc>`_

These packages can be installed using `conda` or `pip`.

To build the documentation in HTML format, execute::

    $ cd docs
    $ make html

This will save the documentation website in the ``../../lightkurve-docs`` directory
on your system.  The notebook-based tutorials will not be recompiled by default
because they take some time to build.  To recompile the notebooks, type::

    make notebooks

Finally, if you have write permission to *lightkurve*'s GitHub repository,
you can upload the documentation to the web server using::

    make upload
