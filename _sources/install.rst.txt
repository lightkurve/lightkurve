.. _installation:

************
Installation
************

Requirements
============

**lightkurve** has the following requirements, all of which tend to be
available by default in a modern installation of Python:

- Python: 2.7, 3.5, 3.6 or later.
- Astropy: 1.3 or later.
- Numpy: 1.11 or later.
- Scipy: 0.19 or later.
- Matplotlib: 1.5.3 or later.
- Astroquery 0.3.7 or later.

We recommend using the `Anaconda Python <https://www.continuum.io/downloads>`_
distribution, which will install Python alongside its most common scientific
packages, including all those listed above.



Installing lightkurve
=====================

Stable version
--------------

The easiest way to install or upgrade lightkurve is with ``pip``,
simply run the following command on a terminal window::

    $ pip install lightkurve --upgrade


.. note::

    The ``--upgrade`` flag is optional, but recommended if you already
    have lightkurve installed and want to upgrade to the latest version.

.. note::

    If you get a ``PermissionError`` this means that you do not have the
    required administrative access to install new packages to your Python
    installation.  In this case you may consider using the ``--user`` option
    to install the package into your home directory.  You can read more
    about how to do this in the `pip documentation
    <http://www.pip-installer.org/en/1.2.1/other-tools.html#using-pip-with-the-user-scheme>`_.


Development version
-------------------

Alternatively, if you want to experiment with the latest development version of
lightkurve, you can install it straight from GitHub::

    $ git clone https://github.com/KeplerGO/lightkurve.git
    $ cd lightkurve
    $ pip install -e .


Building documentation
======================

.. note::

    In general, building the documentation is not necessary unless you are
    writing new documentation or do not have internet access, because the
    latest version of lightkurve's documentation is available online at
    `lightkurve.keplerscience.org/ <http://lightkurve.keplerscience.org/>`_ .

.. note::
    **lightkurve** documentation requires the `numpydoc sphinx extension <https://github.com/numpy/numpydoc>`_
    which can be installed with ``pip install numpydoc``.

To build the documentation to HTML format, you can do::

    cd docs
    make html
