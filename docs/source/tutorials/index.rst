.. _tutorials:

Tutorials
=========


1. The Lightkurve API
---------------------

The first set of tutorials cover the basics of using Lightkurve.
This includes getting to grips with the file types and how to load and work
with data from the Kepler, K2, and TESS missions. For a complete listing of
all classes and methods, please consult the `API docs <../api/index.html>`_.

.. toctree::
    :maxdepth: 1

    01-target-pixel-files.ipynb
    01-what-are-lightcurves.ipynb
    01-lightcurve-files.ipynb
    01-using-the-periodogram-class.ipynb




2. Creating and correcting light curves
---------------------------------------

The second section focuses on the various ways in which light curves
can be extracted from the pixel data, and on the removal of instrument noise
("systematics") from those light curves.

2.1. Creating light curves
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    03-making-custom-apertures.ipynb
    03-how-to-use-prf-photometry.ipynb
    03-cutting-out-tpfs-from-tess-ffis.ipynb
    03-making-fits-files.ipynb
    03-appending-lightcurves.ipynb
    03-using-river-plots.ipynb



2.2. Identifying instrument noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    04-interact-with-lightcurves-and-tpf.ipynb
    04-identify-rolling-band.ipynb


2.3. Removing instrument noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    04-how-to-remove-tess-scattered-light-using-regressioncorrector.ipynb
    04-how-to-detrend.ipynb
    04-removing-cbvs.ipynb




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
