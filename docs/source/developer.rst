.. _developer:

=======================
Developer documentation
=======================

Lightkurve is actively developed on its `GitHub repository <https://github.com/KeplerGO/lightkurve>`_.
This page provides guidelines for package developers and maintainers.


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

Building the *lightkurve* documentation requires a few extra packages:

- sphinx
- sphinx-automodapi
- nbsphinx
- ghp-import
- graphviz
- `numpydoc <https://github.com/numpy/numpydoc>`_

These packages can be installed using `conda` or `pip`.

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


Release procedure
-----------------

The procedure to release a new version of Lightkurve requires a number
of manual steps:

1. Add any new contributors to `AUTHORS.rst`.

2. Remove the suffix `.dev` from the version number in `lightkurve/version.py`. Note that Lightkurve follows a `semantic versioning scheme <https://semver.org>`_.

3. Verify that all unit tests pass:

.. code-block:: bash

    $ pytest --remote-data

4. Verify that the docs and all tutorial notebooks can be compiled without failure (see *Building documentation* above), and upload them to the webserver.

5. Make a new release branch in GitHub using the `Draft a new release button` at https://github.com/KeplerGO/lightkurve/releases.

6. Create and upload the new package to PyPI:

.. code-block:: bash

    $ python setup.py release

7. Increment the version number on the `lightkurve conda feedstock <https://github.com/conda-forge/lightkurve-feedstock>`_.  Navigate to the :code:`meta.yaml` file in the :code:`recipe` directory and change two lines:

.. code-block:: yaml

  {% set version = "1.0b20" %}
  {% set sha256 = "b65f556362b64d9fcc8aa4b0e89991c2b8f7fc059dfa22269b0877a8b758b255" %}

Replace the `version` and `sha256` values with the most recent release info in `the lightkurve PyPI <https://pypi.org/project/lightkurve/>`_.  Submit your update as a Pull Request, which offers a checklist to guide programmatic spot-checking of your code.

8. Edit `lightkurve/version.py` to contain the next version number with suffix `.dev`.
