.. title:: Lightkurve docs

.. rst-class:: frontpage

**********
Lightkurve
**********

A friendly Python package for TESS & Kepler time series analysis.

.. **Version**: |version|

.. raw:: html

    <a href="quickstart.html" class="btn btn-primary">Quickstart →</a>


Lightkurve offers a user-friendly way to analyze time series data obtained by telescopes,
in particular NASA’s TESS and Kepler exoplanet missions.
Lightkurve aims to lower barriers, promote best practices, reduce costs,
and improve scientific fidelity by providing accessible open source
Python :ref:`tools <api>` and :ref:`tutorials <tutorials>`
for time domain astronomy.


.. code-block:: python
    :caption: Example: downloading & plotting a phase-folded light curve of Proxima Centauri obtained by TESS.

    import lightkurve as lk
    lightcurve = lk.search_lightcurve("Proxima Centauri", mission="TESS").download()
    lightcurve.fold(period=11.184).plot()


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    quickstart
    tutorials/index
    reference/index
    about/index
