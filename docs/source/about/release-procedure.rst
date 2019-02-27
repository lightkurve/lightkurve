.. _developer:

Releasing a new version
=======================

This page details the procedure package maintainers should follow to release a new version of Lightkurve.

The procedure requires the following steps:

1. Add any new contributors to `AUTHORS.rst`.

2. Remove the suffix `.dev` from the version number in `lightkurve/version.py` and commit to master. Note that Lightkurve follows a `semantic versioning scheme <https://semver.org>`_.

3. Verify that all unit tests pass:

.. code-block:: bash

    $ pytest --remote-data

4. Verify that the docs and all tutorial notebooks can be compiled without failure (see *Building documentation* above), and upload them to the webserver.

5. Edit the CHANGES.rst file by changing the date for the version you are about to release from “unreleased” to today’s date. Also be sure to make sure the change log is complete and accurate. Then add and commit those changes with:

.. code-block:: bash

    $ git add CHANGES.rst
    $ git commit -m "Finalizing changelog for v<version>"

6. Make a new release branch in GitHub using the `Draft a new release button` at https://github.com/KeplerGO/lightkurve/releases.

7. Create and upload the new package to PyPI:

.. code-block:: bash

    $ python setup.py release

8. Increment the version number on the `lightkurve conda feedstock <https://github.com/conda-forge/lightkurve-feedstock>`_.  Navigate to the :code:`meta.yaml` file in the :code:`recipe` directory and change two lines:

.. code-block:: yaml

  {% set version = "1.0b20" %}
  {% set sha256 = "b65f556362b64d9fcc8aa4b0e89991c2b8f7fc059dfa22269b0877a8b758b255" %}

Replace the `version` and `sha256` values with the most recent release info in `the lightkurve PyPI <https://pypi.org/project/lightkurve/>`_.  Submit your update as a Pull Request, which offers a checklist to guide programmatic spot-checking of your code.

9. Edit `lightkurve/version.py` to contain the next version number with suffix `.dev`.
