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


2.2. Identifying instrument noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    2-creating-light-curves/2-2-kepler-noise-1-data-gaps-and-quality-flags.ipynb
    2-creating-light-curves/2-2-kepler-noise-2-spurious-signals-and-time-sampling-effects.ipynb
    2-creating-light-curves/2-2-kepler-noise-3-seasonal-and-detector-effects.ipynb
    2-creating-light-curves/2-2-kepler-noise-4-electronic-noise.ipynb    
    2-creating-light-curves/2-2-identifying-rolling-band.ipynb


2.3. Removing instrumental noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    04-how-to-use-cbvs.ipynb
    04-how-to-remove-tess-scattered-light-using-regressioncorrector.ipynb
    04-how-to-detrend.ipynb


3. Science examples
-------------------

In the final section we cover some data analysis tasks that astronomers might
commonly want to do with time series data.

3.1. Exoplanet examples
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    02-recover-a-planet.ipynb
    02-how-to-recover-the-first-tess-candidate.ipynb
    05-advanced_patterns_binning.ipynb


3.2. Asteroseismology examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    02-how-to-optimize-periodogram-snr.ipynb
    02-asteroseismology.ipynb


3.3. Other examples
~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    02-how-to-make-a-supernova-lightcurve.ipynb
