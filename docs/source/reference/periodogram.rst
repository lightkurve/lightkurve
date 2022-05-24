.. _api.periodogram:

===========
Periodogram 
===========
.. currentmodule:: lightkurve.periodogram


Constructor
~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  Periodogram
  LombScarglePeriodogram.from_lightcurve
  BoxLeastSquaresPeriodogram.from_lightcurve


Attributes
~~~~~~~~~~

.. autosummary::
  :toctree: api/

  Periodogram.frequency
  Periodogram.frequency_at_max_power
  Periodogram.period
  Periodogram.period_at_max_power
  Periodogram.power
  Periodogram.max_power


Methods
~~~~~~~

.. autosummary::
  :toctree: api/

  Periodogram.bin
  Periodogram.copy
  Periodogram.flatten
  Periodogram.plot
  Periodogram.show_properties
  Periodogram.smooth
  Periodogram.to_seismology
  Periodogram.to_table
  BoxLeastSquaresPeriodogram.compute_stats
  BoxLeastSquaresPeriodogram.get_transit_model
