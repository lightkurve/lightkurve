.. _api.correctors:

======================
Correcting systematics
======================
.. currentmodule:: lightkurve.correctors

Telescope data is always affected by noise contributed by the instrument.
The ``lightkurve.correctors`` sub-package provides classes which offer
different strategies to remove such noise.
At the core of the package lies the generic `.RegressionCorrector` class.
It uses linear regression to correlate a light curve against a `.DesignMatrix`
of column vectors which are known to correlate with additive noise components.

The `CBVCorrector`, `PLDCorrector`, and `SFFCorrector` classes extend `RegressionCorrector`
by providing the user with pre-configured `DesignMatrix` objects which are
known to be effective at removing different types of noise.

Cotrending Basis Vectors (CBV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  load_tess_cbvs
  load_kepler_cbvs
  CBVCorrector
  CBVCorrector.correct
  CBVCorrector.diagnose


Pixel Level Decorrelation (PLD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  PLDCorrector
  PLDCorrector.correct
  PLDCorrector.diagnose
  PLDCorrector.diagnose_masks


Self Flat Fielding (SFF)
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  SFFCorrector
  SFFCorrector.correct
  SFFCorrector.diagnose
  SFFCorrector.diagnose_arclength


Regression Corrector
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  RegressionCorrector
  RegressionCorrector.correct
  RegressionCorrector.diagnose


Creating a design matrix
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  DesignMatrix
  DesignMatrixCollection
  SparseDesignMatrix
  SparseDesignMatrixCollection

A DesignMatrix has the following attributes:

.. autosummary::
  :toctree: api/

  DesignMatrix.X
  DesignMatrix.rank
  DesignMatrix.shape
  DesignMatrix.values

A DesignMatrix supports the following operations:

.. autosummary::
  :toctree: api/

  DesignMatrix.append_constant
  DesignMatrix.collect
  DesignMatrix.copy
  DesignMatrix.pca
  DesignMatrix.plot
  DesignMatrix.plot_priors
  DesignMatrix.split
  DesignMatrix.standardize
  DesignMatrix.to_sparse
  DesignMatrix.validate



.. autosummary::
  :toctree: api/

  corrector.Corrector
  corrector.Corrector.correct
  corrector.Corrector.diagnose
