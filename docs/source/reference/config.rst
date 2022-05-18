.. _api.config:

=============
Configuration
=============
.. currentmodule:: lightkurve


``Lightkurve`` uses ``Astropy``'s configuration system for configurable parameters.

Users can set their defaults in their configuration file, defaulted at ``$HOME/.lightkurve/config/lightkurve.cfg``.

Furthermore, they can also change the values at runtime via `lightkurve.conf` object.

The remaining specifics can be found in `Astropy documentation <https://docs.astropy.org/en/stable/config/index.html>`_.


Access configuration values
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  conf
  config.get_config_dir
