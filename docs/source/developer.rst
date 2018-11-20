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


Release procedure
-----------------

The procedure to release a new version of Lightkurve requires a number
of manual steps:

1. Add any new contributors to `AUTHORS.rst`.

2. Change the version number in `lightkurve/version.py`. Lightkurve follows a `semantic versioning scheme <https://semver.org>`_.

3. Verify that all unit tests pass:

.. code-block:: bash

    $ pytest --remote-data

4. Make a new release branch in GitHub using the `Draft a new release button` at https://github.com/KeplerGO/lightkurve/releases.

5. Create and upload the new package to PyPI:

.. code-block:: bash

    $ python setup.py release

6. Update the online docs:

.. code-block:: bash

    $ cd docs
    $ make clean
    $ make html
    $ make upload

7. Edit `lightkurve/version.py` to contain the next version number with suffix `.dev`.
