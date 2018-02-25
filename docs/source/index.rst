..

======================
Welcome to lightkurve!
======================

The **lightkurve** Python package offers a beautiful and user-friendly way
to analyze astronomical flux time series data,
in particular the pixels and lightcurves obtained by
**NASA's Kepler, K2, and TESS missions**.

.. image:: _static/images/lightkurve-teaser.gif
   :target: _static/images/lightkurve-teaser.gif

This package aims to lower the barrier for both students, astronomers,
and citizen scientists interested in analyzing Kepler and TESS space telescope data.
It does this by providing **high-quality building blocks and tutorials**
which enable both hand-tailored data analyses and advanced automated pipelines.

Lightkurve is an **open source community project** supported by
`NASA's Kepler/K2 Guest Observer Office <https://keplerscience.arc.nasa.gov>`_.
The development `takes place on GitHub <https://github.com/KeplerGO/lightkurve>`_
and everyone is :ref:`invited to contribute<contributing>`.


.. _user-docs:

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   tutorials/quickstart.ipynb
   install
   contributing
   citing
   other_software

.. toctree::
    :caption: Tutorials
    :maxdepth: 1

    How to find a planet using lightkurve? <tutorials/how-to-find-a-kepler-planet-with-lightkurve.ipynb>
    tutorials/target-pixel-files.ipynb
    What are light curve files? <tutorials/lightcurve-files.ipynb>

.. toctree::
    :caption: API Documentation
    :maxdepth: 1
    
    api/targetpixelfile
    api/lightcurve
    api/prf
    api/utils
