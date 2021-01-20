.. _installation:

*********************
Installing Lightkurve
*********************

Using pip
=========

The easiest way to install *Lightkurve* and all of its dependencies is to use the ``pip`` command,
which which is a standard part of all Python distributions.
To install *Lightkurve*, run the following command in a terminal window::

    $ python -m pip install lightkurve --upgrade

The ``--upgrade`` flag is optional, but recommended if you already
have *Lightkurve* installed and want to upgrade to the latest version.

Depending on the specific Python environment, you may need to replace ``python``
with the correct Python interpreter, e.g., ``python3``.


.. note::

    If you encounter a ``PermissionError`` this means that you do not have the
    required administrative access to install new packages to your Python
    installation.  In this case you may consider using the ``--user`` option
    to install the package into your home directory.  You can read more
    about how to do this in the `pip documentation
    <http://www.pip-installer.org/en/1.2.1/other-tools.html#using-pip-with-the-user-scheme>`_.


To verify which version of *Lightkurve* you have installed, run::

    $ python -c "import lightkurve; print(lightkurve.__version__)"



Requirements
============

*Lightkurve* has the following minimum requirements:

- Python: 3.6 or later.
- Astropy: 4.1 or later.
- Numpy: 1.11 or later.
- Scipy: 0.19 or later.
- Matplotlib: 1.5.3 or later.
- Astroquery: 0.3.10 or later.
- Bokeh: 1.0 or later (for interactive widgets).
- Pandas.

If you install *Lightkurve* using ``pip`` as explained above,
these dependencies will be installed automatically if needed.



Installing the development version
==================================

If you want to experiment with the latest development version of
*Lightkurve*, you can install it straight from the master branch on GitHub::

    $ git clone https://github.com/KeplerGO/lightkurve.git
    $ cd lightkurve
    $ python -m pip install .

If you want to have a so-called editable install which enables the installed
version to immediately reflect changes made in the source tree, you can use:

    $ python shim-setup.py develop

Please see our guide on :ref:`contributing to lightkurve<contributing>`
for additional instructions.
