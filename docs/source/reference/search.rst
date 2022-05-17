.. _api.search:

================
Downloading data
================
.. currentmodule:: lightkurve


Searching the Kepler & TESS archive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lightkurve provides three functions which make it easy to search and download
TESS and Kepler data products from the public data archive at MAST.

.. autosummary::
  :toctree: api/

  search_targetpixelfile
  search_lightcurve
  search_tesscut


Downloading data products
~~~~~~~~~~~~~~~~~~~~~~~~~

The search functions listed above return a `SearchResult` object,
which provides an easy way to select and download data.

.. autosummary::
  :toctree: api/

  SearchResult
  SearchResult.download
  SearchResult.download_all


Filtering search results
~~~~~~~~~~~~~~~~~~~~~~~~

The `SearchResult` object provides convenient access to the essential metadata,
which enables the search results to be filtered.
For example, a search result can be filtered by exposure time using
``result = result[result.exptime.value < 100]``.

.. autosummary::
  :toctree: api/

  SearchResult.mission
  SearchResult.year
  SearchResult.author
  SearchResult.target_name
  SearchResult.exptime
  SearchResult.distance
  SearchResult.ra
  SearchResult.dec
  SearchResult.table


Customizing search results display
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can optionally include a list of extra columns in the default display of `SearchResult` objects.

.. autosummary::
  :toctree: api/

  SearchResult.display_extra_columns


Data products collection
~~~~~~~~~~~~~~~~~~~~~~~~

`SearchResult.download_all` returns a `LightCurveCollection` or a
`TargetPixelFileCollection` object. They contain the data products, along with some convenience functions.

A collection can also be further filtered using standard Python list syntax,
and a subset of `numpy.ndarray` syntax.
For example, a collection can be filtered by TESS sectors using
``lcc[(lcc.sector >= 13) & (lcc.sector <= 19)]``.

.. autosummary::
  :toctree: api/

  LightCurveCollection
  LightCurveCollection.stitch
  LightCurveCollection.plot
  LightCurveCollection.append
  LightCurveCollection.campaign
  LightCurveCollection.quarter
  LightCurveCollection.sector
  TargetPixelFileCollection
  TargetPixelFileCollection.plot
  TargetPixelFileCollection.append
  TargetPixelFileCollection.campaign
  TargetPixelFileCollection.quarter
  TargetPixelFileCollection.sector
