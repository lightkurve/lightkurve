.. title:: Lightkurve docs

.. rst-class:: frontpage

**********
Lightkurve
**********

A friendly package for TESS & Kepler time series analysis in Python.

.. **Version**: |version|

.. raw:: html

    <a href="quickstart.html" class="btn btn-primary">Quickstart →</a>

----

**Time domain astronomy made easy for all**

Lightkurve offers a user-friendly way to analyze time series data obtained by telescopes,
in particular NASA’s TESS and Kepler exoplanet missions.

Lightkurve aims to lower barriers, promote best practices, reduce costs,
and improve scientific fidelity by providing accessible
Python :ref:`tools <api>` and :ref:`tutorials <tutorials>`.


.. code-block:: python

    import lightkurve as lk

    pixels = lk.search_targetpixelfile("Kepler-10").download()
    pixels.plot()

    lightcurve = pixels.to_lightcurve()
    lightcurve.plot()

    exoplanet = lightcurve.flatten().fold(period=0.838)
    exoplanet.plot()



.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    quickstart
    tutorials/index
    reference/index
    about/index
