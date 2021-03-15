.. _tutorials:

Tutorials
=========


1. Getting started with Lightkurve
----------------------------------

The first set of tutorials covers the basics of using Lightkurve.
This includes getting to grips with the basic Lightkurve objects
and how to load and work with data products from the Kepler, K2, and TESS missions.
For a complete listing of all classes and methods, please consult the `API docs <../reference/index.html>`_.

1.1. Lightkurve objects
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    1-getting-started/what-are-lightcurve-objects.ipynb
    1-getting-started/what-are-targetpixelfile-objects.ipynb
    1-getting-started/what-are-periodogram-objects.ipynb
    

1.2. Working with Kepler & TESS data products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    1-getting-started/searching-for-data-products.ipynb
    1-getting-started/using-light-curve-file-products.ipynb
    1-getting-started/using-target-pixel-file-products.ipynb
    1-getting-started/plotting-target-pixel-files.ipynb
    1-getting-started/interactively-inspecting-data.ipynb
    1-getting-started/how-to-open-a-lightcurve-in-excel.ipynb


2. Creating and correcting light curves
---------------------------------------

The second section focuses on the various ways in which light curves
can be extracted from the pixel data, and on the removal of instrument noise
(often referred to as *systematics*) from those light curves.

2.1. Creating light curves
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    2-creating-light-curves/2-1-custom-aperture-photometry.ipynb
    2-creating-light-curves/2-1-combining-multiple-quarters.ipynb
    2-creating-light-curves/2-1-cutting-out-tpfs.ipynb
    2-creating-light-curves/2-1-saving-a-light-curve.ipynb


2.2. Identifying instrumental noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    2-creating-light-curves/2-2-kepler-noise-1-data-gaps-and-quality-flags.ipynb
    2-creating-light-curves/2-2-kepler-noise-2-spurious-signals-and-time-sampling-effects.ipynb
    2-creating-light-curves/2-2-kepler-noise-3-seasonal-and-detector-effects.ipynb
    2-creating-light-curves/2-2-kepler-noise-4-electronic-noise.ipynb    
    2-creating-light-curves/2-2-identifying-rolling-band.ipynb
    2-creating-light-curves/2-2-how-to-use-cbvs.ipynb


2.3. Removing instrumental noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    2-creating-light-curves/2-3-how-to-use-cbvcorrector.ipynb
    2-creating-light-curves/2-3-removing-scattered-light-using-regressioncorrector.ipynb
    2-creating-light-curves/2-3-k2-pldcorrector.ipynb
    2-creating-light-curves/2-3-k2-sffcorrector.ipynb


3. Science examples
-------------------

In the final section we demonstrate scientific data analysis tasks which astronomers
commonly apply to time series data.

3.1. Exoplanet examples
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    3-science-examples/exoplanets-identifying-transiting-planet-signals.ipynb
    3-science-examples/exoplanets-recover-a-known-planet.ipynb
    3-science-examples/exoplanets-recover-first-tess-candidate.ipynb
    3-science-examples/exoplanets-machine-learning-preprocessing.ipynb
    3-science-examples/exoplanets-visualizing-periodic-signals-using-a-river-plot.ipynb

3.2. Rotation rates and periodic signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    3-science-examples/periodograms-creating-periodograms.ipynb
    3-science-examples/periodograms-measuring-a-rotation-period.ipynb
    3-science-examples/periodograms-verifying-the-location-of-a-signal.ipynb
    3-science-examples/periodograms-optimizing-the-snr.ipynb

3.3. Asteroseismology examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    3-science-examples/asteroseismology-estimating-mass-and-radius.ipynb
    3-science-examples/asteroseismology-oscillating-star-periodogram.ipynb

3.4. Other examples
~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    3-science-examples/other-supernova-lightcurve.ipynb
