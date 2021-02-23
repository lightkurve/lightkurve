.. _api.seismology:

==========
Seismology 
==========
.. currentmodule:: lightkurve.seismology


Constructor
~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  Seismology
  Seismology.from_lightcurve

Attributes
~~~~~~~~~~

.. autosummary::
  :toctree: api/

  Seismology.periodogram


Estimating parameters
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  Seismology.estimate_numax
  Seismology.estimate_deltanu
  Seismology.estimate_radius
  Seismology.estimate_mass
  Seismology.estimate_logg


Visualizing the results
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  Seismology.diagnose_numax
  Seismology.diagnose_deltanu
  Seismology.plot_echelle
  Seismology.interact_echelle
