.. _api:

API
===

LightCurve objects
------------------

LightCurve objects contain time-series data on the brightness of a star. They provide easy access to a range of operations, e.g. folding, binning, plotting, and a variety of signal processing. Light curves observed by Kepler or TESS have a specific sub-class which provide extra metadata, but the generic LightCurve object can be used for any data set.

.. automodsumm:: lightkurve.lightcurve


Opening data files
------------------

The lightkurve.search modules makes it easy to open data products from Kepler and TESS, and search for them at the data archive.

.. automodsumm:: lightkurve.search


LightCurveFile objects represent files that are used to store LightCurves and their metadata. Files of this type are found at NASA’s data archives.

.. automodsumm:: lightkurve.lightcurvefile


TargetPixelFile objects hold the sequence of images (pixels) which can be converted into LightCurve objects using different techniques.

.. automodsumm:: lightkurve.targetpixelfile


Utility objects and functions
-----------------------------

.. automodsumm:: lightkurve.utils
.. automodsumm:: lightkurve.periodogram



Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`