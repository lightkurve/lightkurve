.. _api.lightcurve:

==========
LightCurve
==========
.. currentmodule:: lightkurve

The `LightCurve` class is an extension of AstroPy's `~astropy.timeseries.TimeSeries`
object, which itself extends AstroPy's `~astropy.table.Table`.
Compared to a generic Table, LightCurve objects enforce the presence of three
special data columns: `~LightCurve.time`, `~LightCurve.flux`, and `~LightCurve.flux_err`.
This enables a LightCurve object to offer a range of methods which are specific to working
with flux-based time series data.

Constructor
~~~~~~~~~~~

Light curves can be instantiated by passing array-like values to the
``time``, ``flux``, and ``flux_err`` parameters.
Additional columns can be added by passing a table-like object to the ``data`` parameter.

.. autosummary::
   :toctree: api/

   LightCurve


Attributes
~~~~~~~~~~

Another difference with AstroPy `~astropy.table.Table` is that all columns can
be accessed conveniently as attributes.  For example, ``LightCurve.time`` is
offered as a shorthand for ``LightCurve["time"]``.

.. autosummary::
   :toctree: api/

   LightCurve.time
   LightCurve.flux
   LightCurve.flux_err


Metadata
~~~~~~~~

All meta data are stored in the `meta` dictionary.
For convenience, meta data can be accessed as object attributes,
e.g. ``LightCurve.sector`` is a short-hand for ``LightCurve.meta["SECTOR"]``.

.. autosummary::
   :toctree: api/

   LightCurve.meta


Plotting
~~~~~~~~

.. autosummary::
  :toctree: api/

  LightCurve.plot
  LightCurve.scatter
  LightCurve.errorbar
  LightCurve.plot_river
  MPLSTYLE


Data manipulation
~~~~~~~~~~~~~~~~~

The following methods all return a new `LightCurve` object.

.. autosummary::
  :toctree: api/

  LightCurve.append
  LightCurve.copy
  LightCurve.bin
  LightCurve.fill_gaps
  LightCurve.flatten
  LightCurve.fold
  LightCurve.head
  LightCurve.normalize
  LightCurve.remove_nans
  LightCurve.remove_outliers
  LightCurve.select_flux
  LightCurve.tail
  LightCurve.truncate



Conversions
~~~~~~~~~~~
.. autosummary::
  :toctree: api/

  LightCurve.to_corrector
  LightCurve.to_csv
  LightCurve.to_excel
  LightCurve.to_fits
  LightCurve.to_pandas
  LightCurve.to_periodogram
  LightCurve.to_seismology
  LightCurve.to_table
  LightCurve.write



Other Utility Methods
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  LightCurve.estimate_cdpp
  LightCurve.query_solar_system_objects
  LightCurve.interact_bls
  LightCurve.create_transit_mask
  LightCurve.search_neighbors



