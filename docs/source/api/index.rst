.. _api:

API
===

This page provides an overview of the Python functions and classes provided by the Lightkurve package.

Programmers tend to refer to a list like this as the API (*Application Programming Interface*).


Light curve objects
-------------------

The :class:`lightkurve.lightcurve` module provides classes which represent time-series brightness (flux) data.
The generic :class:`LightCurve <lightkurve.lightcurve.LightCurve>` class can be used to work with any "time vs flux" data set and provides access to a range of common operations, e.g.
`~lightkurve.lightcurve.LightCurve.bin()`,
`~lightkurve.lightcurve.LightCurve.flatten()`,
`~lightkurve.lightcurve.LightCurve.fold()`,
and `~lightkurve.lightcurve.LightCurve.plot()`.

.. automodsumm:: lightkurve.lightcurve
	:skip: KeplerLightCurve, TessLightCurve


Searching and downloading data
------------------------------

The :class:`lightkurve.search` module provides functions which make it easy to load and search data files produced by the Kepler and TESS missions.

.. automodsumm:: lightkurve.search
    :skip: SearchResult, search_lightcurvefile

The :class:`lightkurve.io` module provides functions to load data files produced by the Kepler and TESS missions.

.. automodsumm:: lightkurve.io
    :skip: open

Data product objects
--------------------

The :class:`lightkurve.targetpixelfile` module provides classes which represent
FITS files that store the original pixel data (images) obtained by the Kepler
or TESS telescopes. These classes provide methods to visualize these data and
extract custom light curves.

.. automodsumm:: lightkurve.targetpixelfile



Correcting systematics
----------------------

Telescope data is always affected by systematic noise contributed by the detector.
The :class:`lightkurve.correctors` sub-package provides classes which offer
different strategies to remove such noise.
At the core of the package lies the generic `.RegressionCorrector` class.
It uses linear regression to correlate a light curve against a `.DesignMatrix`
of column vectors which are known to correlate with additive noise components:

.. automodsumm:: lightkurve.correctors
    :skip: SFFCorrector, PLDCorrector, KeplerCBVCorrector, TessPLDCorrector

The classes below extend `.RegressionCorrector` by providing the user with
pre-configured `.DesignMatrix` objects which are known to be effective at
removing different types of noise:

.. automodsumm:: lightkurve.correctors
    :skip: RegressionCorrector, DesignMatrix, DesignMatrixCollection



Finding periodic signals
------------------------

The :class:`lightkurve.periodogram` module provides classes to help find periodic signals in light curves.

.. automodsumm:: lightkurve.periodogram


Asteroseismology
----------------

The :class:`lightkurve.seismology` sub-package provides tools to extra quick-look astroseismic parameters (numax, deltanu, radius, mass, and logg) from periodograms.

.. automodsumm:: lightkurve.seismology
    :skip: estimate_deltanu_acf2d, diagnose_deltanu_acf2d, estimate_numax_acf2d, diagnose_numax_acf2d, estimate_radius, estimate_mass, estimate_logg


Utilities
---------

The :class:`lightkurve.utils` module provides a range of common helper functions and classes.

.. automodsumm:: lightkurve.utils
    :skip: LightkurveWarning, bkjd_to_astropy_time, btjd_to_astropy_time


Can't find what you're looking for?
-----------------------------------

If you are looking for a specific class or function not listed here, try consulting the API index or search pages:

* :ref:`Index of all classes and methods <genindex>`
* :ref:`Index of all modules <modindex>`
* :ref:`Search page <search>`