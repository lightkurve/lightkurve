Lightkurve
==========

**A friendly package for Kepler & TESS time series analysis in Python.**

**Documentation: https://docs.lightkurve.org**

|test-badge| |conda-badge| |pypi-badge| |pypi-downloads| |doi-badge| |astropy-badge|

.. |conda-badge| image:: https://img.shields.io/conda/vn/conda-forge/lightkurve.svg
                 :target: https://anaconda.org/conda-forge/lightkurve
.. |pypi-badge| image:: https://img.shields.io/pypi/v/lightkurve.svg
                :target: https://pypi.python.org/pypi/lightkurve
.. |pypi-downloads| image:: https://pepy.tech/badge/lightkurve
                :target: https://pepy.tech/project/lightkurve
.. |test-badge| image:: https://github.com/lightkurve/lightkurve/workflows/Lightkurve-tests/badge.svg
                 :target: https://github.com/lightkurve/lightkurve/actions?query=branch%3Amain
.. |astropy-badge| image:: https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
                   :target: http://www.astropy.org
.. |doi-badge| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1181928.svg
              :target: https://docs.lightkurve.org/about/citing.html             

**Lightkurve** is a community-developed, open-source Python package which offers a beautiful and user-friendly way
to analyze astronomical flux time series data,
in particular the pixels and lightcurves obtained by
**NASA's Kepler and TESS exoplanet missions**.

.. image:: https://raw.githubusercontent.com/lightkurve/lightkurve/main/docs/source/_static/images/lightkurve-teaser.gif

This package aims to lower the barrier for students, astronomers,
and citizen scientists interested in analyzing Kepler and TESS space telescope data.
It does this by providing **high-quality building blocks and tutorials**
which enable both hand-tailored data analyses and advanced automated pipelines.


Documentation
-------------

Read the documentation at `https://docs.lightkurve.org <https://docs.lightkurve.org>`_.


Quickstart and Installation
---------------------------

Please visit our quickstart guide at `https://docs.lightkurve.org/quickstart.html <https://docs.lightkurve.org/quickstart.html>`_. 

The easiest way to install *Lightkurve* and all of its dependencies is to use the ``pip`` command,
which is a standard part of all Python distributions.
To install *Lightkurve*, run the following command in a terminal window::

    $ python -m pip install lightkurve --upgrade

The ``--upgrade`` flag is optional, but recommended if you already
have *Lightkurve* installed and want to upgrade to the latest version.

Depending on the specific Python environment, you may need to replace ``python``
with the correct Python interpreter, e.g., ``python3``.

If you want to experiment with the latest development version of
*Lightkurve*, you can install it straight from the main branch on GitHub:

.. code-block:: bash

    $ git clone https://github.com/lightkurve/lightkurve.git
    $ cd lightkurve
    $ python -m pip install .

If you want to have a so-called editable install which enables the installed
version to immediately reflect changes made in the source tree, you can use:

.. code-block:: bash

    $ python -m pip install poetry
    $ poetry install

Please see our guide on `https://docs.lightkurve.org/development/index.html <https://docs.lightkurve.org/development/index.html>`_
for additional instructions.


Contributing
------------

We welcome community contributions!
Please read the  guidelines at `https://docs.lightkurve.org/development/contributing.html <https://docs.lightkurve.org/development/contributing.html>`_.


Citing
------

If you find Lightkurve useful in your research, please cite it and give us a GitHub star!
Please read the citation instructions at `https://docs.lightkurve.org/about/citing.html <https://docs.lightkurve.org/about/citing.html>`_.


Contact
-------
Lightkurve is an open source community project created by `the authors <AUTHORS.rst>`_.
The best way to contact us is to `open an issue <https://github.com/lightkurve/lightkurve/issues/new>`_ or to e-mail tesshelp@bigbang.gsfc.nasa.gov.
Please include a self-contained example that fully demonstrates your problem or question.
