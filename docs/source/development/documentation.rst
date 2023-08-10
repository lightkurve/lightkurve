.. _docs:

Building the documentation
==========================

Coding and documentation standards
----------------------------------

Lightkurve adopts AstroPy's coding guidelines and standards,
as documented in `AstroPy's Development Documentation <http://docs.astropy.org/en/stable/index.html#developer-documentation>`_.


Building documentation
----------------------

.. note::

    Building the documentation is not necessary unless you are
    writing new documentation or do not have internet access, because the
    latest version of the documentation is available online at
    `docs.lightkurve.org <https://docs.lightkurve.org/>`_ .

Building the *lightkurve* documentation requires `sphinx` and a few extra packages. We recommend using `poetry` to install the development dependencies::

    $ poetry install

To make a clean directory for the docs use::

    $ cd docs
    $ make clean

To build the documentation in HTML format, execute::

    $ cd docs
    $ make html

Note if you build the documentation after cleaning the directory this will compile the notebooks, which can take a significant amount of time (over 30 minutes). This will save the documentation website in the ``../../lightkurve-docs`` directory
on your system.  If you re-run the `make html` command the notebook-based tutorials will not be recompiled by default
because they take some time to build.  To recompile the just the notebooks, type::

    $ make notebooks

Finally, if you have write permission to *lightkurve*'s GitHub repository,
you can upload the documentation to the web server using::

    $ make upload

.. note::

    If you encounter the error ``Pandoc wasn't found``, you will have to install ``pandoc`` separately as well following its `installation instruction <https://pandoc.org/installing.html>`_  .
    For example, on Mac OS you can install ``pandoc`` using ``homebrew``::

        $ brew install pandoc

    An alternative method is to install pandoc using ``conda``::

        $ conda install --channel=conda-forge pandoc

.. note::
    
    To build the docs on a Mac you may have to install the Xcode Command Line Tools, which you can do using::

        $ xcode-select --install
    



