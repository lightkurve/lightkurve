"""Provides tools to model a Kepler image for PRF photometry fitting.

Example use
-----------
%matplotlib inline
import numpy as np
from lightkurve import KeplerTargetPixelFile, LightCurve
from lightkurve.prf import StarPrior, BackgroundPrior, FocusPrior, MotionPrior, TPFModel, PRFPhotometry
from oktopus import GaussianPrior, UniformPrior

tpf = KeplerTargetPixelFile("https://archive.stsci.edu/missions/kepler/target_pixel_files/0084/008462852/"
                            "kplr008462852-2013098041711_lpd-targ.fits.gz", quality_mask='hardest')

# First, compute a few values from our TPF which will inform the priors
bgflux = np.nanpercentile(tpf.flux[0], 10)
maxflux = np.nansum(tpf.flux, axis=(1, 2)).max()
col, row = np.nanmedian(tpf.centroids(), axis=1)

# Set up the model
model = TPFModel(star_priors=[StarPrior(col=GaussianPrior(mean=col, var=2**2),
                                        row=GaussianPrior(mean=row, var=2**2),
                                        flux=UniformPrior(lb=0, ub=maxflux),
                                        targetid=tpf.keplerid)],
                 background_prior=BackgroundPrior(flux=GaussianPrior(mean=bgflux, var=bgflux)),
                 focus_prior=FocusPrior(scale_col=GaussianPrior(mean=1, var=0.0001),
                                        scale_row=GaussianPrior(mean=1, var=0.0001),
                                        rotation_angle=UniformPrior(lb=-3.1415, ub=3.1415)),
                 motion_prior=MotionPrior(shift_col=GaussianPrior(mean=0., var=0.01),
                                          shift_row=GaussianPrior(mean=0., var=0.01)),
                 prfmodel=tpf.get_prf_model(),
                 fit_background=True,
                 fit_focus=False,
                 fit_motion=False)

pp = PRFPhotometry(model)
pp.run(tpf.flux, pos_corr1=tpf.pos_corr1, pos_corr2=tpf.pos_corr2, cadences=range(1650, 1850))
pp.plot_results()
print('The star flux in the first cadence is {}'.format(pp.results[0].stars[0].flux))
"""
from __future__ import division, print_function

import logging
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings

from oktopus import Prior, GaussianPrior, UniformPrior, PoissonPosterior

from .prfmodel import KeplerPRF
from ..utils import plot_image


__all__ = ['GaussianPrior', 'UniformPrior', 'FixedValuePrior',
           'StarPrior', 'BackgroundPrior', 'FocusPrior', 'MotionPrior',
           'StarParameters', 'BackgroundParameters', 'FocusParameters',
           'MotionParameters', 'TPFModelParameters',
           'TPFModel', 'PRFPhotometry']


log = logging.getLogger(__name__)


class FixedValuePrior(Prior):
    """An improper prior with a negative log probability of 0 at a single fixed
    value and inf elsewhere. This is similar to a Dirac Delta function,
    except this function does not peak at infinity so that it can be used
    in numerical optimization functions.  It does not integrate to one as a
    result and is therefore an "improper distribution".

    Attributes
    ----------
    value : int or array-like of ints
        The fixed value.

    Examples
    --------
    >>> fp = FixedValuePrior(1)
    >>> fp(1)
    -0.0
    >>> fp(0.5)
    inf
    """
    def __init__(self, value, name=None):
        self.value = np.asarray([value]).reshape(-1)
        self.name = name

    def __repr__(self):
        return "<FixedValuePrior(value={})>".format(self.value)

    @property
    def mean(self):
        """Returns the fixed value."""
        return self.value

    @property
    def variance(self):
        """Returns zero."""
        return 0

    def evaluate(self, params):
        """Returns the negative log pdf."""
        if self.value == params:
            return -0.0
        return np.inf

    def gradient(self, params):
        raise NotImplementedError()


class PriorContainer(object):
    """Container object to hold parameter priors for PRF photometry."""
    def _parse_prior(self, prior):
        if isinstance(prior, Prior):
            return prior
        return FixedValuePrior(value=prior)

    def __call__(self, *params):
        """Calls :func:`evaluate`"""
        return self.evaluate(*params)


class StarPrior(PriorContainer):
    """Container class to capture a user's beliefs about a star's position and flux.

    Example use
    -----------
    StarPrior(col=GaussianPrior(mean=col, var=err_col**2),
              row=GaussianPrior(mean=row, var=err_row**2),
              flux=GaussianPrior(mean=flux, var=err_flux**2))
    """
    def __init__(self, col, row, flux=UniformPrior(lb=0, ub=1e10), targetid=None):
        self.col = self._parse_prior(col)
        self.row = self._parse_prior(row)
        self.flux = self._parse_prior(flux)
        self.targetid = targetid

    def __repr__(self):
        return ('<StarPrior(\n  col={}\n  row={}\n  flux={}\n  targetid={})>'
                ''.format(self.col, self.row, self.flux, self.targetid))

    def evaluate(self, col, row, flux):
        """Evaluate the prior probability of a star of a given flux being at
        a given row and col.
        """
        logp = (self.col.evaluate(col) +
                self.row.evaluate(row) +
                self.flux.evaluate(flux))
        return logp


class BackgroundPrior(PriorContainer):
    """Container class to capture a user's beliefs about the background flux.

    Parameters
    ----------
    flux : oktopus ``Prior`` object
        Prior on the background flux in electrons/second per pixel.
    """
    def __init__(self, flux=FixedValuePrior(value=0)):
        self.flux = self._parse_prior(flux)

    def __repr__(self):
        return ('<BackgroundPrior(\n  flux={})>'.format(self.flux))

    def evaluate(self, flux):
        """Returns the prior probability for a given background flux value."""
        return self.flux.evaluate(flux)


class FocusPrior(PriorContainer):
    """Container class to capture a user's beliefs about the telescope focus.

    Parameters
    ----------
    scale_col, scale_row : oktopus ``Prior`` object
        Pixel scale in the column and row directions. Typically close to one.
    rotation_angle : oktopus ``Prior`` object
        Rotation angle in radians. Typically zero.
    """
    def __init__(self,
                 scale_col=GaussianPrior(mean=1, var=0.0001),
                 scale_row=GaussianPrior(mean=1, var=0.0001),
                 rotation_angle=UniformPrior(lb=-3.1415, ub=3.1415)):
        self.scale_col = self._parse_prior(scale_col)
        self.scale_row = self._parse_prior(scale_row)
        self.rotation_angle = self._parse_prior(rotation_angle)

    def __repr__(self):
        return ('<FocusPrior(\n  scale_col={}\n  scale_row={}\n  rotation_angle={})>'
                ''.format(self.scale_col, self.scale_row, self.rotation_angle))

    def evaluate(self, scale_col, scale_row, rotation_angle):
        """Returns the prior probability for a gien set of focus parameters."""
        logp = (self.scale_col.evaluate(scale_col) +
                self.scale_row.evaluate(scale_row) +
                self.rotation_angle.evaluate(rotation_angle))
        return logp


class MotionPrior(PriorContainer):
    """Container class to capture a user's beliefs about the telescope motion.
    """
    def __init__(self, shift_col=GaussianPrior(mean=0, var=1.**2),
                 shift_row=GaussianPrior(mean=0, var=1.**2)):
        self.shift_col = self._parse_prior(shift_col)
        self.shift_row = self._parse_prior(shift_row)

    def __repr__(self):
        return ('<MotionPrior(\n  shift_col={}\n  shift_row={})>'
                ''.format(self.shift_col, self.shift_row))

    def evaluate(self, shift_col, shift_row):
        """Returns the prior probability for a gien set of motion parameters."""
        logp = (self.shift_col.evaluate(shift_col) +
                self.shift_row.evaluate(shift_row))
        return logp


class StarParameters(object):
    """Container class to hold the parameters of a star in a ``TPFModel``.
    """
    def __init__(self, col, row, flux, err_col=None, err_row=None, err_flux=None,
                 targetid=None):
        self.col = col
        self.row = row
        self.flux = flux
        self.targetid = targetid

    def __repr__(self):
        r = "<StarParameters(\n  col={}\n  row={}\n  flux={}\n  targetid={})>".format(
                    self.col, self.row, self.flux, self.targetid)
        return r


class BackgroundParameters(object):
    """Container class to hold the parameters of the background in a ``TPFModel``.
    """
    def __init__(self, flux=0., err_flux=None, fitted=True):
        self.flux = flux
        self.err_flux = err_flux
        self.fitted = fitted

    def __repr__(self):
        r = "<BackgroundParameters(\n  flux={}\n  fitted={})>".format(
                    self.flux, self.fitted)
        return r


class FocusParameters(object):
    """Container class to hold the parameters of the telescope focus in a ``TPFModel``.
    """
    def __init__(self, scale_col=1., scale_row=1., rotation_angle=0., fitted=False):
        self.scale_col = scale_col
        self.scale_row = scale_row
        self.rotation_angle = rotation_angle
        self.fitted = fitted

    def __repr__(self):
        return ("<FocusParameters(\n  scale_col={}\n  scale_row={}\n  "
                "rotation_angle={}\n  fitted={})>"
                "".format(self.scale_col, self.scale_row,
                          self.rotation_angle, self.fitted))


class MotionParameters(object):
    """Container class to hold the parameters of the telescope motion in a ``TPFModel``.
    """
    def __init__(self, shift_col=0., shift_row=0., fitted=False):
        self.shift_col = shift_col
        self.shift_row = shift_row
        self.fitted = fitted

    def __repr__(self):
        return "<MotionParameters(\n  shift_col={}\n  shift_row={}\n  fitted={})>".format(
                    self.shift_col, self.shift_row, self.fitted)


class TPFModelParameters(object):
    """Container class to combine all parameters that parameterize a ``TPFModel``.

    Attributes
    ----------
    stars : list of ``StarParameters`` objects
        Parameters related to the stars in the model.
    background : ``BackgroundParameters`` object
        Parameters related to the background flux.
    focus : ``FocusParameters`` object
        Parameters related to the telescope focus.
    motion : ``MotionParameters`` object
        Parameters related to the telescope motion.
    """
    def __init__(self, stars=[], background=BackgroundParameters(),
                 focus=FocusParameters(), motion=MotionParameters()):
        self.stars = stars
        self.background = background
        self.focus = focus
        self.motion = motion

    def __repr__(self):
        out = super(TPFModelParameters, self).__repr__() + '\n'
        out += ''.join(['  {}\n'.format(str(star).replace('\n', '\n  '))
                        for star in self.stars])
        out += '  ' + str(self.background).replace('\n', '\n  ') + '\n'
        out += '  ' + str(self.focus).replace('\n', '\n  ') + '\n'
        out += '  ' + str(self.motion).replace('\n', '\n  ') + '\n'
        if 'residual_image' in vars(self):
            out += '  residual_image:\n    {}'.format(self.residual_image[0][0:4])[:-1]
            out += '...\n'
        if 'predicted_image' in vars(self):
            out += '  predicted_image:\n    {}'.format(self.predicted_image[0][0:4])[:-1]
            out += '...\n'
        return out

    def to_array(self):
        """Converts the free parameters held by this class to an array of size (n,),
        where n is the number of free parameters.

        This method exists because `scipy.optimize` can only optimize arrays of
        real numbers, yet we like to store in the parameters in human-friendly
        container classes to make the fitted parameters accessible without
        confusion.

        Returns
        -------
        array : array-like
            Array containing all the free parameters.
        """
        array = []
        for star in self.stars:
            array.append(star.col)
            array.append(star.row)
            array.append(star.flux)
        if self.background.fitted:
            array.append(self.background.flux)
        if self.focus.fitted:
            array.append(self.focus.scale_col)
            array.append(self.focus.scale_row)
            array.append(self.focus.rotation_angle)
        if self.motion.fitted:
            array.append(self.motion.shift_col)
            array.append(self.motion.shift_row)
        return np.array(array).ravel()

    def from_array(self, array):
        """Inverse of ``to_array()``."""
        array = np.asarray(array).ravel()
        next_idx = 0
        stars = []
        for staridx in range(len(self.stars)):
            star = StarParameters(col=array[next_idx],
                                  row=array[next_idx + 1],
                                  flux=array[next_idx + 2])
            stars.append(star)
            next_idx += 3

        if not self.background.fitted:
            background = self.background
        else:
            background = BackgroundParameters(flux=array[next_idx], fitted=True)
            next_idx += 1

        if not self.focus.fitted:
            focus = self.focus
        else:
            focus = FocusParameters(scale_col=array[next_idx],
                                    scale_row=array[next_idx + 1],
                                    rotation_angle=array[next_idx + 2],
                                    fitted=True)
            next_idx += 3

        if not self.motion.fitted:
            motion = self.motion
        else:
            motion = MotionParameters(shift_col=array[next_idx],
                                      shift_row=array[next_idx + 1],
                                      fitted=True)

        return TPFModelParameters(stars=stars, background=background,
                                  focus=focus, motion=motion)


class TPFModel(object):
    """A model which describes a single-cadence Kepler image.

    Attributes
    ----------
    star_priors : list of ``StarPrior`` objects.
        List of stars believed to be in the image.
    background_prior : ``BackgroundPrior`` object.
        Beliefs about the background flux.
    prfmodel : ``KeplerPRF`` object.
        The callable Pixel Reponse Function (PRF) model to use.
    focus_prior : ``FocusPrior`` object.
        Beliefs about the telescope focus.
    motion_prior : ``MotionPrior`` object.
        Beliefs about the telescope motion.
    fit_background : bool
        If False, the background parameters will be kept fixed.
    fit_focus : bool
        If False, the telescope focus parameters will be kept fixed.
    fit_motion : bool
        If False, the telescope motion parameters will be kept fixed.
    """
    def __init__(self, star_priors=[],
                 background_prior=BackgroundPrior(),
                 focus_prior=FocusPrior(),
                 motion_prior=MotionPrior(),
                 prfmodel=None,
                 fit_background=True, fit_focus=False, fit_motion=False):
        if prfmodel is None:
            prfmodel = KeplerPRF(1, shape=(10, 10), column=0, row=0)
        self.star_priors = star_priors
        self.background_prior = background_prior
        self.focus_prior = focus_prior
        self.motion_prior = motion_prior
        self.prfmodel = prfmodel
        self.fit_background = fit_background
        self.fit_focus = fit_focus
        self.fit_motion = fit_motion
        self._params = self.get_initial_guesses()

    def __repr__(self):
        out = super(TPFModel, self).__repr__() + '\n'
        out += ''.join(['  {}\n'.format(str(star).replace('\n', '\n  '))
                        for star in self.star_priors])
        out += '  ' + str(self.background_prior).replace('\n', '\n  ') + '\n'
        out += '  ' + str(self.focus_prior).replace('\n', '\n  ') + '\n'
        out += '  ' + str(self.motion_prior).replace('\n', '\n  ') + '\n'
        out += '  ' + str(self.prfmodel).replace('\n', '\n  ') + '\n'
        out += '  fit_background={}\n  fit_focus={}\n  fit_motion={}\n'.format(
                        self.fit_background, self.fit_focus, self.fit_motion)
        return out

    def get_initial_guesses(self):
        """Returns the prior means which can be used to initialize the model.

        The guesses are obtained by taking the means of the priors.
        """
        initial_star_guesses = []
        for star in self.star_priors:
            initial_star_guesses.append(StarParameters(col=star.col.mean,
                                                       row=star.row.mean,
                                                       flux=star.flux.mean))
        background = BackgroundParameters(flux=self.background_prior.flux.mean,
                                          fitted=self.fit_background)
        focus = FocusParameters(scale_col=self.focus_prior.scale_col.mean,
                                scale_row=self.focus_prior.scale_row.mean,
                                rotation_angle=self.focus_prior.rotation_angle.mean,
                                fitted=self.fit_focus)
        motion = MotionParameters(shift_col=self.motion_prior.shift_col.mean,
                                  shift_row=self.motion_prior.shift_row.mean,
                                  fitted=self.fit_motion)
        initial_params = TPFModelParameters(stars=initial_star_guesses,
                                            background=background,
                                            focus=focus,
                                            motion=motion)
        return initial_params

    def predict(self, params=None):
        """Returns a synthetic Kepler image given a set of model parameters.

        Attributes
        ----------
        params : ```TPFModelParameters``` object
            Parameters which define the model.

        Returns
        -------
        synthetic_image : 2D ndarray
            Predicted image given the parameters.
        """
        if params is None:
            params = self.get_initial_guesses()
        star_images = []
        for star in params.stars:
            star_images.append(self.prfmodel(center_col=star.col + params.motion.shift_col,
                                             center_row=star.row + params.motion.shift_row,
                                             flux=star.flux,
                                             scale_col=params.focus.scale_col,
                                             scale_row=params.focus.scale_row,
                                             rotation_angle=params.focus.rotation_angle))
        synthetic_image = np.sum(star_images, axis=0) + params.background.flux
        return synthetic_image

    def _predict(self, *params_array):
        """Wrapper around ``predict()`` which takes an array of shape (n,)
        where n is the number of free parameters.

        Unlike ``predict()`, this function can be called by scipy.optimize.
        """
        params = self._params.from_array(params_array)
        return self.predict(params)

    def __call__(self, *params_array):
        return self._predict(*params_array)

    def gradient(self, *params_array):
        """UNFINISHED WORK!"""
        params = self._params.from_array(params_array)
        grad = []
        for star in params.stars:
            grad.append(self.prfmodel.gradient(center_col=star.col,
                                               center_row=star.row,
                                               flux=star.flux))
        # We assume the background gradient is proportional to one
        grad.append([np.ones(self.prfmodel.shape)])
        # We assume the gradient of other parameters is one
        for i in range(len([params_array]) - 3 * len(params.stars) - 1):
            grad.append([np.ones(self.prfmodel.shape)])
        grad = sum(grad, [])
        return grad

    def logp_prior(self, params):
        """Evaluates the prior at a point in the parameter space.

        Attributes
        ----------
        params : TPFModelParameters object
        """
        logp = 0
        for star, star_prior in zip(params.stars, self.star_priors):
            logp += star_prior.evaluate(col=star.col, row=star.row, flux=star.flux)
        if self.fit_background:
            logp += self.background_prior.evaluate(params.background.flux)
        if self.fit_focus:
            logp += self.focus_prior.evaluate(params.focus.scale_col,
                                              params.focus.scale_row,
                                              params.focus.rotation_angle)
        if self.fit_motion:
            logp += self.motion_prior.evaluate(params.motion.shift_col,
                                               params.motion.shift_row)
        return logp

    def _logp_prior(self, params_array):
        """Wrapper around ``logp_prior()`` which takes an array of shape (n,)
        where n is the number of free parameters.

        Unlike ``predict()`, this function can be called by scipy.optimize.
        """
        params = self._params.from_array(params_array)
        return self.logp_prior(params)

    def fit(self, data, loss_function=PoissonPosterior, method='powell',
            pos_corr1=None, pos_corr2=None, **kwargs):
        """Fits the model to the data.

        Parameters
        ----------
        data : array-like
            The pixel data for a single cadence, i.e. the data obtained using
            ``KeplerTargetPixelFile.flux[cadenceno]``.
        loss_function : subclass of oktopus.LossFunction
            Noise distribution associated with each random measurement
        kwargs : dict
            Dictionary of additional parameters to be passed to
            `scipy.optimize.minimize`.

        Returns
        -------
        result : ``TPFModelParameters`` object
            Fitted parameters plus fitting diagnostics.
        """
        if pos_corr1 is not None and np.abs(pos_corr1) < 50:
            self.motion_prior.shift_col.mean = pos_corr1
        if pos_corr2 is not None and np.abs(pos_corr2) < 50:
            self.motion_prior.shift_row.mean = pos_corr2

        self._params = self.get_initial_guesses()  # Update _params for model changes!
        loss = loss_function(data, self, prior=self._logp_prior)
        with warnings.catch_warnings():
            # Ignore RuntimeWarnings trigged by invalid values
            warnings.simplefilter("ignore", RuntimeWarning)
            fit = loss.fit(x0=self.get_initial_guesses().to_array(), method=method, **kwargs)
        result = self._params.from_array(fit.x)
        # NOTE: uncertainties are not available for now because `self.gradient` is unfinished;
        # hence, the line below is commented out for now:
        # result.uncertainties = loss.loglikelihood.uncertainties(fit.x)
        result.predicted_image = self._predict(fit.x)
        result.residual_image = data - result.predicted_image
        result.loss_value = fit.fun
        result.opt_result = fit
        return result

    def plot(self, *params, **kwargs):
        """Plots an image of the model for a given point in the parameter space."""
        img = self.predict(*params)
        plot_image(img,
                   title='TPF Model',
                   extent=(self.prfmodel.column, self.prfmodel.column + self.prfmodel.shape[1],
                           self.prfmodel.row, self.prfmodel.row + self.prfmodel.shape[0]),
                   **kwargs)

    def plot_diagnostics(self, data, figsize=(12, 4), *params, **kwargs):
        """Plots an image of the model for a given point in the parameter space."""
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        fit = self.fit(data)
        plot_image(data, ax=ax[0],
                   title='Observed Data, Channel: {}'.format(self.prfmodel.channel),
                   extent=(self.prfmodel.column, self.prfmodel.column + self.prfmodel.shape[1],
                           self.prfmodel.row, self.prfmodel.row + self.prfmodel.shape[0]),
                   **kwargs)
        plot_image(fit.predicted_image, ax=ax[1],
                   title='Predicted Image, Channel: {}'.format(self.prfmodel.channel),
                   extent=(self.prfmodel.column, self.prfmodel.column + self.prfmodel.shape[1],
                           self.prfmodel.row, self.prfmodel.row + self.prfmodel.shape[0]),
                   **kwargs)
        plot_image(fit.residual_image, ax=ax[2],
                   title='Residual Image, Channel: {}'.format(self.prfmodel.channel),
                   extent=(self.prfmodel.column, self.prfmodel.column + self.prfmodel.shape[1],
                           self.prfmodel.row, self.prfmodel.row + self.prfmodel.shape[0]),
                   **kwargs)
        return fit


class PRFPhotometry(object):
    """This class performs PRF Photometry on TPF-like data given a ``TPFModel``.

    This class exists because a ``TPFModel`` object is designed to fit only
    one cadence at a time.  This class makes it easy to fit a large number
    of cadences and obtain the resulting LightCurve.

    Attributes
    ----------
    model : instance of TPFModel
        Model which will be fit to the data
    """
    def __init__(self, model):
        self.model = model
        self.results = []

    def run(self, tpf_flux, cadences=None, pos_corr1=None, pos_corr2=None, parallel=True):
        """Fits the model to the flux data.

        Parameters
        ----------
        tpf_flux : array-like
            A pixel flux time-series, i.e., the pixel data, e.g,
            KeplerTargetPixelFile.flux, such that (time, row, column) represents
            the shape of ``tpf_flux``.
        cadences : array-like
            Cadences to fit.  If `None` (default) then all cadences will be fit.
        pos_corr1, pos_corr2 : array-like, array-like
            If set, use these values to update the prior means for
            `model.motion_prior.shift_col` and `model.motion_prior.shift_row`
            for each cadence.
        parallel : boolean
            If `True`, cadences will be fit in parallel using Python's `multiprocessing` module. 
        """
        if cadences is None:  # By default, fit all cadences.
            cadences = np.arange(len(tpf_flux))
        # Prepare an iterable of arguments, such that each item contains all information
        # needed to fit a single cadence.  This will enable parallel processing below.
        tpf_flux = np.asarray(tpf_flux)  # Ensure the flux data can be indexed
        if pos_corr1 is None or pos_corr2 is None:
            args = zip([self.model]*len(cadences),
                      tpf_flux[cadences])
        else:
            args = zip([self.model]*len(cadences),
                      tpf_flux[cadences],
                       pos_corr1[cadences],
                       pos_corr2[cadences])            
        # Set up a mapping function
        if parallel:
            import multiprocessing
            pool = multiprocessing.Pool()
            mymap = pool.imap
        else:
            import itertools
            mymap = itertools.imap
        # Now fit all cadences using the mapping function and the list of arguments
        self.results = []
        for result in tqdm(mymap(fit_one_cadence, args), desc='Fitting cadences', total=len(cadences)):
            self.results.append(result)
        if parallel:
            pool.close()
        # Parse results
        self.lightcurves = [self._parse_lightcurve(star_idx)
                            for star_idx in range(len(self.model.star_priors))]

    def _parse_lightcurve(self, star_idx):
        # Create a lightcurve
        from .. import LightCurve
        flux = []
        for cadence in range(len(self.results)):
            flux.append(self.results[cadence].stars[star_idx].flux)
        return LightCurve(flux=flux, targetid=self.model.star_priors[star_idx].targetid)

    def _parse_background(self):
        # Create a lightcurve
        from .. import LightCurve
        bgflux = []
        for cadence in range(len(self.results)):
            bgflux.append(self.results[cadence].background.flux)
        return LightCurve(flux=bgflux)

    def plot_results(self, star_idx=0):
        """Plot all the TPF model parameters over time."""
        fig, ax = plt.subplots(10, sharex=True, figsize=(6, 12))
        x = range(len(self.results))
        ax[0].plot(x, [r.stars[star_idx].flux for r in self.results])
        ax[0].set_ylabel('Flux')
        ax[1].plot(x, [r.stars[star_idx].col for r in self.results])
        ax[1].set_ylabel('Col')
        ax[2].plot(x, [r.stars[star_idx].row for r in self.results])
        ax[2].set_ylabel('Row')
        ax[3].plot(x, [r.motion.shift_col for r in self.results])
        ax[3].set_ylabel('Shift col')
        ax[4].plot(x, [r.motion.shift_row for r in self.results])
        ax[4].set_ylabel('Shift row')
        ax[5].plot(x, [r.background.flux for r in self.results])
        ax[5].set_ylabel('Background')
        ax[6].plot(x, [r.focus.scale_col for r in self.results])
        ax[6].set_ylabel('Focus col')
        ax[7].plot(x, [r.focus.scale_row for r in self.results])
        ax[7].set_ylabel('Focus row')
        ax[8].plot(x, [r.focus.rotation_angle for r in self.results])
        ax[8].set_ylabel('Focus angle')
        ax[9].plot(x, [r.loss_value for r in self.results])
        ax[9].set_ylabel('Loss')


def fit_one_cadence(arg):
    """Helper function to enable parallelism.

    This function is used by PRFPhotometry.run().
    """
    model = arg[0]
    data = arg[1]
    if len(arg) == 4:
        pos_corr1, pos_corr2 = arg[2], arg[3]
    else:
        pos_corr1, pos_corr2 = None, None
    return model.fit(data, pos_corr1=pos_corr1, pos_corr2=pos_corr2)
