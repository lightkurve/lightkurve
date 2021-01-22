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
``result = result[result.t_exptime < 100]``.

.. autosummary::
  :toctree: api/

  SearchResult.observation
  SearchResult.author
  SearchResult.target_name
  SearchResult.t_exptime
  SearchResult.productFilename
  SearchResult.distance
  SearchResult.ra
  SearchResult.dec
  SearchResult.table
