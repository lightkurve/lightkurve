"""Defines PLDCorrector.

This module requires 3 optional dependencies (theano, pymc3, exoplanet) for
PLD correction to work.  One additional dependency (fbpca) is not required
but will speed up the computation if installed.

TODO
----
* Make sure the prior of logs2/logsigma/logrho has a user-configurable width.
* [kinda done] Fix the units of corrected_flux, corrected_flux_without_gp, etc.
  In Lightkurve we currently add back the mean of the model.
* The design matrix can be improved by rejecting pixels which are saturated,
  and including the collapsed sums of their CCD columns instead.
* Set PyMC verbosity to match the lightkurve.log.level.
* It is not clear whether the inclusion of a column vector of ones in the
  design matrix is necessary for numerical stability.
"""
import logging
from itertools import combinations_with_replacement as multichoose

import numpy as np
import matplotlib.pyplot as plt

# Optional dependencies
try:
    import pymc3 as pm
    import exoplanet as xo
    import theano.tensor as tt
except ImportError:
    # Fail quietly here so we don't break `import lightkurve`.
    # We will raise a user-friendly ImportError inside PLDCorrector.__init__().
    pass

from .. import MPLSTYLE

log = logging.getLogger(__name__)

__all__ = ['PLDCorrector']


class PLDCorrector(object):
    r"""Implements the Pixel Level Decorrelation (PLD) systematics removal method.

        Pixel Level Decorrelation (PLD) was developed by [1]_ to remove
        systematic noise caused by spacecraft jitter for the Spitzer
        Space Telescope. It was adapted to K2 data by [2]_ and [3]_
        for the EVEREST pipeline [4]_.

        For a detailed description and implementation of PLD, please refer to
        these references. Lightkurve provides a reference implementation
        of PLD that is less sophisticated than EVEREST, but is suitable
        for quick-look analyses and detrending experiments.

        Our implementation of PLD is performed by first calculating the noise
        model for each cadence in time. This function goes up to arbitrary
        order, and is represented by

        .. math::

            m_i = \sum_l a_l \frac{f_{il}}{\sum_k f_{ik}} + \sum_l \sum_m b_{lm} \frac{f_{il}f_{im}}{\left( \sum_k f_{ik} \right)^2} + ...
        where

          - :math:`m_i` is the noise model at time :math:`t_i`
          - :math:`f_{il}` is the flux in the :math:`l^\text{th}` pixel at time :math:`t_i`
          - :math:`a_l` is the first-order PLD coefficient on the linear term
          - :math:`b_{lm}` is the second-order PLD coefficient on the :math:`l^\text{th}`,
            :math:`m^\text{th}` pixel pair

        We perform Principal Component Analysis (PCA) to reduce the number of
        vectors in our final model to limit the set to best capture instrumental
        noise. With a PCA-reduced set of vectors, we can construct a design matrix
        containing fractional pixel fluxes.

        To capture long-term variability, we simultaneously fit a Gaussian Process
        model ([5]_) to the underlying stellar signal. We use the gradient-based
        probabilistic modeling toolkit [6]_ to optimize the GP hyperparameters and
        solve for the motion model.

        To robustly estimate errors on our parameter estimates and output flux
        values, we optionally sample the output model with [7]_ and infer errors from
        the posterior distribution.

        To solve for the PLD model, we need to minimize the difference squared

        .. math::

            \chi^2 = \sum_i \frac{(y_i - m_i)^2}{\sigma_i^2},

        where :math:`y_i` is the observed flux value at time :math:`t_i`, by solving

        .. math::

            \frac{\partial \chi^2}{\partial a_l} = 0.

    References
    ----------
    .. [1] Deming et al. (2015), ads:2015ApJ...805..132D.
        (arXiv:1411.7404)
    .. [2] Luger et al. (2016), ads:2016AJ....152..100L
        (arXiv:1607.00524)
    .. [3] Luger et al. (2018), ads:2018AJ....156...99L
        (arXiv:1702.05488)
    .. [4] EVEREST pipeline webpage, https://rodluger.github.io/everest
    .. [5] Celerite documentation, https://celerite.readthedocs.io/en/stable/
    .. [6] Exoplanet documentation, https://exoplanet.readthedocs.io/en/stable/
    .. [7] PyMC3 documentation, https://docs.pymc.io

    Parameters
    ----------
    tpf : TargetPixelFile
    aperture_mask : 2D boolean array or str
    pld_aperture_mask : 2D boolean array or str

    Raises
    ------
    ImportError : if one of Lightkurve's optional dependencies required by
        this class are missing.
    """
    def __init__(self, tpf, aperture_mask=None, pld_aperture_mask=None):
        # Ensure the optional dependencies requires by this class are installed
        success, messages = self._check_optional_dependencies()
        if not success:
            for message in messages:
                log.error(message)
            raise ImportError("\n".join(messages))

        # Input validation: parse the aperture masks to accept strings etc.
        self.aperture_mask = tpf._parse_aperture_mask(aperture_mask)
        self.pld_aperture_mask = tpf._parse_aperture_mask(pld_aperture_mask)
        # Generate raw flux light curve from desired pixels
        raw_lc = tpf.to_lightcurve(aperture_mask=self.aperture_mask)
        # It is critical to remove all NaNs or the linear algebra below will crash
        self.raw_lc, self.nan_mask = raw_lc.remove_nans(return_mask=True)
        self.tpf = tpf[~self.nan_mask]
        # For user-friendliness, store most recent solutions
        self.most_recent_model = None
        self.most_recent_solution_or_trace = None

    def _check_optional_dependencies(self):
        """Emits a user-friendly error message if one of Lightkurve's
        optional dependencies which are required for PLDCorrector are missing.

        Returns
        -------
        success : bool
            True if all optional dependencies are available, False otherwise.
        message : list of str
            User-friendly error message if success == False.
        """
        success, messages = True, []

        try:
            import pymc3 as pm
        except ImportError:
            success = False
            messages.append("PLDCorrector requires pymc3 to be installed (`pip install pymc3`).")

        try:
            import exoplanet as xo
        except ImportError:
            success = False
            messages.append("PLDCorrector requires exoplanet to be installed (`pip install exoplanet`).")

        try:
            import theano.tensor as tt
        except ImportError:
            success = False
            messages.append("PLDCorrector requires theano to be installed (`pip install theano`).")

        return success, messages

    def create_first_order_matrix(self):
        """Returns a matrix which encodes the fractional pixel fluxes as a function
        of cadence (row) and pixel (column). As such, the method returns a
        2D matrix with shape (n_cadences, n_pixels_in_pld_mask).

        This matrix will form the basis of the PLD regressor design matrix
        and is often called the first order component. The matrix returned
        here is guaranteed to be free of NaN values.

        Returns
        -------
        matrix : numpy array
            First order PLD design matrix.
        """
        # Re-arrange the cube of flux values observed in a user-specified mask
        # into a 2D matrix of shape (n_cadences, n_pixels_in_mask).
        # Note that Theano appears to require 64-bit floats.
        matrix = np.asarray(self.tpf.flux[:, self.pld_aperture_mask], np.float64)
        assert matrix.shape == (len(self.raw_lc.time), self.pld_aperture_mask.sum())
        # Remove all NaN or Inf values
        matrix = matrix[:, np.isfinite(matrix).all(axis=0)]
        # To ensure that each column contains the fractional pixel flux,
        # we divide by the sum of all pixels in the same cadence.
        # This is an important step, as explained in Section 2 of Luger et al. (2016).
        matrix = matrix / np.sum(matrix, axis=-1)[:, None]
        # If we return matrix at this point, theano will raise a "dimension mismatch".
        # The origin of this bug is not understood, but copying the matrix
        # into a new one as shown below circumvents it:
        result = np.empty((matrix.shape[0], matrix.shape[1]))
        result[:, :] = matrix[:, :]

        return result

    def create_design_matrix(self, pld_order=1, n_pca_terms=10, include_column_of_ones=False, **kwargs):
        """Returns a matrix designed to contain suitable regressors for the
        systematics noise model.

        The design matrix contains one row for each cadence (i.e. moment in time)
        and one column for each regressor that we wish to use to predict the
        systematic noise in a given cadence.

        The columns (i.e. regressors) included in the design matrix are:
        * One column for each pixel in the PLD aperture mask.  Each column
          contains the flux values observed by that pixel over time.  This is
          also known as the first order component.
        * Columns derived from the products of all combinations of pixel values
          in the aperture mask. However, rather than including a column for each
          combination, we perform dimensionality reduction (PCA) and include a
          smaller number of PCA terms, i.e. the number of columns is
          n_pca_terms*(pld_order-1).  This is also known as the higher order
          components.
        * Optionally, a single column of ones for numerical stability.

        Thus, the shape of the design matrix will be
        (n_cadences, n_pld_mask_pixels + n_pca_terms*(pld_order-1) + include_column_of_ones)

        Returns
        -------
        design_matrix : 2D numpy array
            See description above.
        """
        # We use an optional dependency for very fast PCA (fbpca), but if the
        # import fails we will fall back on using the slower `np.linalg.svd`.
        use_fbpca = True
        try:
            from fbpca import pca
        except ImportError:
            use_fbpca = False
            log.warning("PLD systematics correction will run faster if the "
                        "optional `fbpca` package is installed "
                        "(`pip install fbpca`).")

        matrix_sections = []  # list to hold the design matrix components
        first_order_matrix = self.create_first_order_matrix()

        # Input validation: n_pca_terms cannot be larger than the number of cadences
        n_cadences = len(first_order_matrix)
        if n_pca_terms > n_cadences:
            log.warning("`n_pca_terms` ({}) cannot be larger than the number of cadences ({});"
                        "using n_pca_terms={}".format(n_pca_terms, n_cadences, n_cadences))
            n_pca_terms = n_cadences

        # The original EVEREST paper includes a column vector of ones in the
        # design matrix to improve the numerical stability (see Luger et al.);
        # it is unclear whether this is necessary, so this is an optional step
        # for now.
        if include_column_of_ones:
            matrix_sections.append([np.ones((len(first_order_matrix), 1))])

        # Add the first order matrix
        matrix_sections.append(first_order_matrix)

        # Add the higher order PLD design matrix columns
        for order in range(2, pld_order + 1):
            # Take the product of all combinations of pixels; order=2 will
            # multiply all pairs of pixels, order=3 will multiple triples, etc.
            matrix = np.product(list(multichoose(first_order_matrix.T, order)), axis=1).T
            # This product matrix becomes very big very quickly, so we reduce
            # its dimensionality using PCA.
            if use_fbpca:  # fast mode
                components, _, _ = pca(matrix, n_pca_terms)
            else:  # slow mode
                components, _, _ = np.linalg.svd(matrix)
            section = components[:, :n_pca_terms]
            matrix_sections.append(section)

        return np.concatenate(matrix_sections, axis=1)

    def create_pymc_model(self, design_matrix=None, cadence_mask=None,
                          gp_timescale_prior=150, **kwargs):
        r"""Returns a PYMC3 model.

        Parameters
        ----------
        design_matrix : np.ndarray
            Matrix of shape (n_cadences, n_regressors) used to create the
            motion model.
        cadence_mask : np.ndarray
            Boolean array to mask cadences. Cadences that are False will be excluded
            from the model fit
        gp_timescale_prior : int
            The parameter `rho` in the definition of the Matern-3/2 kernel, which
            influences the timescale of variability fit by the Gaussian Process.
            For more information, see [1]

        Returns
        -------
        model : pymc3.model.Model
            A pymc3 model.

        References
        ----------
        .. [1] the `celerite` documentation https://celerite.readthedocs.io
        """
        if design_matrix is None:
            design_matrix = self.create_design_matrix(**kwargs)
        if cadence_mask is None:
            cadence_mask = np.ones(len(self.raw_lc.time), dtype=bool)

        # Theano raises an `AsTensorError` ("Cannot convert to TensorType")
        # if the floats are not in 64-bit format, so we convert them here:
        lc_time = np.asarray(self.raw_lc.time, np.float64)
        lc_flux = np.asarray(self.raw_lc.flux, np.float64)
        lc_flux_err = np.asarray(self.raw_lc.flux_err, np.float64)

        # Covariance matrix diagonal
        diag = lc_flux_err**2

        # The cadence mask is applied by inflating the uncertainties in the covariance matrix;
        # this is because celerite will run much faster if it is able to predict
        # data for cadences that have been fed into the model.
        diag[~cadence_mask] += 1e12

        with pm.Model() as model:
            # Create a Gaussian Process to model the long-term stellar variability
            # log(sigma) is the amplitude of variability, estimated from the raw flux scatter
            logsigma = pm.Normal("logsigma", mu=np.log(np.std(lc_flux)), sd=4)
            # log(rho) is the timescale of variability with a user-defined prior
            logrho = pm.Normal("logrho", mu=np.log(gp_timescale_prior), sd=4)
            # log(s2) is a jitter term to compensate for underestimated flux errors
            logs2 = pm.Normal("logs2", mu=np.log(np.var(lc_flux)), sd=4)
            kernel = xo.gp.terms.Matern32Term(log_sigma=logsigma, log_rho=logrho)

            # Store the GP and cadence mask to aid debugging
            model.gp = xo.gp.GP(kernel, lc_time, diag + tt.exp(logs2))
            model.cadence_mask = cadence_mask

            # The motion model regresses against the design matrix
            A = tt.dot(design_matrix.T, model.gp.apply_inverse(design_matrix))
            B = tt.dot(design_matrix.T, model.gp.apply_inverse(lc_flux[:, None]))
            weights = tt.slinalg.solve(A, B)
            motion_model = pm.Deterministic("motion_model", tt.dot(design_matrix, weights)[:, 0])
            pm.Deterministic("weights", weights)

            # Likelihood to optimize
            pm.Potential("obs", model.gp.log_likelihood(lc_flux - motion_model))

            # Add deterministic variables for easy of use
            gp_model, gp_model_var = model.gp.predict(return_var=True)
            pm.Deterministic("gp_model", gp_model)
            pm.Deterministic("gp_model_std", np.sqrt(gp_model_var))
            pm.Deterministic("corrected_flux", lc_flux - motion_model + tt.mean(motion_model))
            pm.Deterministic("corrected_flux_without_gp", lc_flux - motion_model - gp_model + tt.mean(motion_model))

        self.most_recent_model = model
        return model

    def optimize(self, model=None, start=None, robust=False, **kwargs):
        """Returns the maximum likelihood solution.

        Parameters
        ----------
        model : pymc3.model.Model
            A pymc3 model
        start : dict
            MAP Solution from exoplanet
        robust : bool
            If `True`, all parameters will be optimized separately before
            attempting to optimize all parameters together.  This will be
            significantly slower but increases the likelihood of success.
        **kwargs : dict
            Dictionary of arguments to be passed to
            `lightkurve.correctors.PLDCorrector.create_pymc_model`.

        Returns
        -------
        map_soln : dict
            Maximum likelihood values.
        """
        if model is None:
            model = self.create_pymc_model(**kwargs)
        if start is None:
            start = model.test_point

        with model:
            map_soln = xo.optimize(start=start, vars=[model.logs2])
            # Optimizing parameters separately appears to make finding a solution more likely
            if robust:
                map_soln = xo.optimize(start=map_soln, vars=[model.logsigma])
                map_soln = xo.optimize(start=map_soln, vars=[model.logrho, model.logsigma])
                map_soln = xo.optimize(start=map_soln, vars=[model.logs2])
            map_soln = xo.optimize(start=map_soln)  # Optimize all parameters

        self.most_recent_solution_or_trace = map_soln
        return map_soln

    def sample(self, model=None, start=None, draws=1000):
        """Sample the systematics correction model.

        Parameters
        ----------
        model : pymc3.model.Model
            A pymc3 model
        start : dict
            MAP Solution from exoplanet
        draws : int
            Number of samples

        Returns
        -------
        trace : `pymc3.trace`
            Trace object containing parameters and their samples
        """
        # Create the model
        if model is None:
            model = self.create_pymc_model()
        if start is None:
            start = self.optimize()

        # Initialize the sampler
        sampler = xo.PyMC3Sampler()
        with model:
            # Burn in the sampler
            sampler.tune(tune=np.max([int(draws*0.3), 150]),
                         start=start,
                         step_kwargs=dict(target_accept=0.9),
                         chains=4)
            # Sample the parameters
            trace = sampler.sample(draws=draws, chains=4)

        self.most_recent_solution_or_trace = trace
        return trace

    def correct(self, remove_gp_trend=False, sample=False, **kwargs):
        """Returns a systematics-corrected light curve.

        Parameters
        ----------
        remove_gp_trend : boolean
            `True` will subtract the fit GP stellar signal from the returned
            flux light curve.
        sample : boolean
            Boolean expression: `True` will sample the output of the optimization
            step and include robust errors on the output light curve.
        draws : int
            Number of samples.

        Returns
        -------
        corrected_lc : `~lightkurve.lightcurve.LightCurve`
            Motion noise corrected light curve object.
        """
        solution_or_trace = self._compute(**kwargs)
        if remove_gp_trend:
            column_name = 'corrected_flux_without_gp'
        else:
            column_name = 'corrected_flux'
        lc = self._lightcurve_from_solution(solution_or_trace,
                                            column_name=column_name)
        return lc

    def _compute(self, model=None, sample=False, **kwargs):
        """Helper function to compute output solution or trace.

        Parameters
        ----------
        model : pymc3.model.Model
            A pymc3 model
        sample : boolean
            Boolean expression: `True` will sample the output of the optimization
            step and include robust errors on the output light curve.
        **kwargs : dict
            Dictionary of arguments to be passed to
            `lightkurve.correctors.PLDCorrector.create_pymc_model`.

        Returns
        -------
        solution_or_trace : dict or `pymc3.backends.base.MultiTrace`
            Dictionary containing output of `exoplanet` optimization, or outoput
            trace from sampling the solution.
        """
        if model is None:
            model = self.create_pymc_model(**kwargs)
        solution = self.optimize(model=model, **kwargs)
        if sample:
            solution_or_trace = self.sample(model=model, start=solution, **kwargs)
        else:
            solution_or_trace = solution

        return solution_or_trace

    def _lightcurve_from_solution(self, solution_or_trace, column_name='corrected_flux'):
        """Helper function to generate light curve objects from a trace.
        Parameters
        ----------
        solution_or_trace : dict or `pymc3.backends.base.MultiTrace`
            Dictionary containing output of `exoplanet` optimization, or outoput
            trace from sampling the solution.
        sample : boolean
            Boolean expression: `True` will sample the output of the optimization
            step and include robust errors on the output light curve.
        column_name : str
            Key for determining which light curve to extract from the given
            solution or trace.

        Returns
        -------
        lc : `~lightkurve.lightcurve.LightCurve`
            Light curve object corresponding to the input 'column_name'
        """
        lc = self.raw_lc.copy()
        # If a trace is given, find the mean and std
        if isinstance(solution_or_trace, pm.backends.base.MultiTrace):
            lc.flux = np.nanmean(solution_or_trace[column_name], axis=0)
            lc.flux_err = np.nanstd(solution_or_trace[column_name], axis=0)
        # Otherwise, read value from the dictionary
        else:
            lc.flux = solution_or_trace[column_name]
        return lc

    def get_diagnostic_lightcurves(self, solution_or_trace=None, **kwargs):
        """Return useful diagnostic light curves.

        Parameters
        ----------
        solution_or_trace : dict or `pymc3.backends.base.MultiTrace`
            Dictionary containing output of `exoplanet` optimization, or outoput
            trace from sampling the solution.
        **kwargs : dict
            Dictionary of arguments to be passed to
            `lightkurve.correctors.PLDCorrector.optimize`.

        Returns
        -------
        corrected_lc : `~lightkurve.lightcurve.LightCurve`
            Motion noise corrected light curve object.
        gp_lc : `~lightkurve.lightcurve.LightCurve`
            Light curve object containing GP model of the stellar signal.
        motion_lc : `~lightkurve.lightcurve.LightCurve`
            Light curve object with the motion model removed by the corrector.
        """
        if solution_or_trace is None:
            if self.most_recent_solution_or_trace is not None:
                solution_or_trace = self.most_recent_solution_or_trace
            else:
                solution_or_trace = self._compute(**kwargs)

        corrected_lc = self._lightcurve_from_solution(solution_or_trace=solution_or_trace,
                                                      column_name='corrected_flux')
        gp_lc = self._lightcurve_from_solution(solution_or_trace=solution_or_trace,
                                               column_name='gp_model')
        motion_lc = self._lightcurve_from_solution(solution_or_trace=solution_or_trace,
                                                   column_name='motion_model')

        return corrected_lc, gp_lc, motion_lc

    def plot_diagnostics(self, solution=None, **kwargs):
        """Plots a series of useful figures to help understand the noise removal
        process.

        Parameters
        ----------
        solution : dict
            Dictionary containing output of `exoplanet` optimization.
        **kwargs : dict
            Dictionary of arguments to be passed to
            `lightkurve.correctors.PLDCorrector.optimize`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """

        # Generate diagnostic light curves
        corrected_lc, gp_lc, motion_lc = self.get_diagnostic_lightcurves(solution, **kwargs)

        # Increase the GP light curve flux normalization from 0 to the motion
        # model mean for visual comparison purposes
        gp_lc.flux = gp_lc.flux + np.nanmean(motion_lc.flux)

        fig, ax = plt.subplots(3, sharex=True, figsize=(8.485, 10))
        # Plot the corrected light curve over the raw flux
        self.raw_lc.scatter(c='r', alpha=0.3, ax=ax[0], label='Raw Flux', normalize=False)
        corrected_lc.scatter(c='k', ax=ax[0], label='Corrected Flux', normalize=False)
        # Plot the stellar model over the raw flux, indicating masked cadences
        self.raw_lc.scatter(c='r', alpha=0.3, ax=ax[1], label='Raw Flux', normalize=False)
        gp_lc.plot(c='k', ax=ax[1], label='GP Model', normalize=False)
        gp_lc[~self.most_recent_model.cadence_mask].scatter(ax=ax[1], label='Masked Cadences', marker='d', normalize=False)
        # Plot the motion model over the raw light curve
        self.raw_lc.scatter(c='r', alpha=0.3, ax=ax[2], label='Raw Flux', normalize=False)
        motion_lc.scatter(c='k', ax=ax[2], label='Noise Model', normalize=False)

        return ax

    def plot_design_matrix(self, design_matrix=None, **kwargs):
        """Plots the design matrix.

        Parameters
        ----------
        design_matrix : np.ndarray
            Matrix of shape (n_cadences, n_regressors) used to create the
            motion model.
        **kwargs : dict
            Dictionary of arguments to be passed to
            `lightkurve.correctors.PLDCorrector.create_design_matrix`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if design_matrix is None:
            design_matrix = self.create_design_matrix(**kwargs)

        with plt.style.context(MPLSTYLE):
            fig, ax = plt.subplots(1, figsize=(8.45, 8.45))
            ax.imshow(design_matrix, aspect='auto')
            ax.set_ylabel('Cadence Number')
            ax.set_xlabel('Regressors')

        return ax
