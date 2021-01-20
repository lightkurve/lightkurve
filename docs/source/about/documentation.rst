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

.. note::

    If you encounter the error ``Pandoc wasn't found``, you will have to install ``pandoc`` separately as well following its `installation instruction <https://pandoc.org/installing.html>`_  .
    For example, on Mac OS you can install ``pandoc`` using ``homebrew``::

        $ brew install pandoc

    An alternative method is to install pandoc using ``conda``::

        $ conda install --channel=conda-forge pandoc

To build the documentation in HTML format, execute::

    $ cd docs
    $ make clean
    $ make html

This will save the documentation website in the ``../../lightkurve-docs`` directory
on your system.  The notebook-based tutorials will not be recompiled by default
because they take some time to build.  To recompile the notebooks, type::

    make notebooks

Finally, if you have write permission to *lightkurve*'s GitHub repository,
you can upload the documentation to the web server using::

    make upload
