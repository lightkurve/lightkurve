"""Draft implementation of a dream Kepler PSF photometry API."""
import numpy as np
from tqdm import tqdm

from oktopus import GaussianPrior, UniformPrior, PoissonPosterior

from .utils import plot_image
from . import LightCurve
from . import KeplerTargetPixelFile


class StarPrior(object):
    """Captures the user's beliefs about a single star's position and flux.

    Example use
    -----------
    StarPrior(col=GaussianPrior(mean=col, var=err_col**2),
              row=GaussianPrior(mean=row, var=err_row**2),
              flux=GaussianPrior(mean=flux, var=err_flux**2))
    """
    def __init__(self, col, row, flux=UniformPrior(lb=0, ub=1e10), targetid=None):
        self.col = col
        self.row = row
        self.flux = flux
        self.targetid = targetid

    def evaluate(self, col, row, flux):
        """Evaluate the prior probability of a star of a given flux being at
        a given row and col.
        """
        logp = (self.col.evaluate(col) +
                self.row.evaluate(row) +
                self.flux.evaluate(flux))
        return logp


class FocusPrior(object):
    def __init__(self, scale_col=GaussianPrior(mean=1, var=0.0001),
                 scale_row=GaussianPrior(mean=1, var=0.0001),
                 rotation_angle=UniformPrior(lb=-3.1415, ub=3.1415)):
        self.scale_col = scale_col
        self.scale_row = scale_row
        self.rotation_angle = rotation_angle

    def evaluate(self, scale_col, scale_row, rotation_angle):
        """Evaluate the prior probability.
        """
        logp = (self.scale_col.evaluate(scale_col) +
                self.scale_row.evaluate(scale_row) +
                self.rotation_angle.evaluate(rotation_angle))
        return logp  


class StarParameters(object):
    """Captures the parameters of a star in a scene model.
    """
    def __init__(self, col, row, flux, err_col=None, err_row=None, err_flux=None):
        self.col = col
        self.row = row
        self.flux = flux


class BackgroundParameters(object):
    """Captures the parameters of the background in a Kepler scene model.
    """
    def __init__(self, flux, err_flux=None, fixed=False):
        self.flux = flux
        self.err_flux = err_flux
        self.fixed = fixed


class FocusParameters(object):
    """Captures the parameters of the telescope focus."""
    def __init__(self, scale_col=1., scale_row=1., rotation_angle=0., fixed=False):
        self.scale_col = scale_col
        self.scale_row = scale_row
        self.rotation_angle = rotation_angle
        self.fixed = fixed


class SceneModelParameters():
    """Parameters that define a single cadence of a TPF image.

    Attributes
    ----------
    stars : list of `StarParameter` objects
        Stars in the scene.
    """
    def __init__(self, stars, background, focus):
        self.stars = stars
        self.background = background
        self.focus = focus        

    def to_array(self):
        """Converts the parameters to an array of real elements of size (n,),
        where n is the number of parameters.

        We do this because scipy.optimize can only optimize arrays of real
        numbers.
        """
        array = []
        for star in self.stars:
            array.append(star.col)
            array.append(star.row)
            array.append(star.flux)
        if not self.background.fixed:
            array.append(self.background.flux)
        if not self.focus.fixed:
            array.append(self.focus.scale_col)
            array.append(self.focus.scale_row)
            array.append(self.focus.rotation_angle)
        return np.array(array)

    def from_array(self, array):
        """Converts an array of parameters to a more human-friendly
        `SceneModelParameters class`.
        """
        next_idx = 0
        stars = []
        for staridx in range(len(self.stars)):
            star = StarParameters(col=array[staridx * 3],
                                  row=array[staridx * 3 + 1],
                                  flux=array[staridx * 3 + 2])
            stars.append(star)
            next_idx = staridx * 3 + 3
        if not self.background.fixed:
            background = BackgroundParameters(flux=array[next_idx])
            next_idx += 1
        else:
            background = self.background
        if not self.focus.fixed:
            focus = FocusParameters(scale_col=array[next_idx],
                                    scale_row=array[next_idx + 1],
                                    rotation_angle=array[next_idx + 2])
        else:
            focus = self.focus
        return SceneModelParameters(stars=stars, background=background, focus=focus)


class SceneModel():
    """A model which describes a single-cadence Kepler image.

    Attributes
    ----------
    star_priors : list of `StarPrior` objects.
        List of stars believed to be in the image.
    background_prior : BackgroundPrior object.
        Beliefs about the per-pixel background flux.
    prfmodel : KeplerPRF object.
    """
    def __init__(self, star_priors, background_prior, prfmodel,
                 focus_prior=None, fix_background=False, fix_focus=True):
        self.star_priors = star_priors
        self.background_prior = background_prior
        self.prfmodel = prfmodel
        self.focus_prior = focus_prior
        if focus_prior is None:
            self.focus_prior = FocusPrior()
        self.fix_background = fix_background
        self.fix_focus = fix_focus
        self.params = self.initial_guesses()

    def initial_guesses(self):
        """Returns the prior means."""
        initial_star_guesses = []
        for star in self.star_priors:
            initial_star_guesses.append(StarParameters(col=star.col.mean,
                                                       row=star.row.mean,
                                                       flux=star.flux.mean))
        background = BackgroundParameters(flux=self.background_prior.mean,
                                          fixed=self.fix_background)
        focus = FocusParameters(scale_col=self.focus_prior.scale_col.mean,
                                scale_row=self.focus_prior.scale_row.mean,
                                rotation_angle=self.focus_prior.rotation_angle.mean,
                                fixed=self.fix_focus)
        initial_params = SceneModelParameters(stars=initial_star_guesses,
                                              background=background,
                                              focus=focus)
        return initial_params

    def predict(self, params=None):
        """Produces a synthetic Kepler 2D image given a set of scene parameters.

        Attributes
        ----------
        params : SceneModelParameters object
        """
        if params is None:
            params = self.initial_guesses()
        star_images = []
        fcs = params.focus
        for star in params.stars:
            star_images.append(self.prfmodel(star.flux, star.col, star.row,
                                             fcs.scale_col, fcs.scale_row, fcs.rotation_angle))
        synthetic_image = np.sum(star_images, axis=0) + params.background.flux
        return synthetic_image

    def _predict(self, *params_array):
        """Version of `predict()` that takes an array instead of a
        SimpleSceneModelParameters object.
        """
        params = self.params.from_array(params_array)
        return self.predict(params)

    def logp_prior(self, params):
        """Evaluates the prior at a point in the parameter space.

        Attributes
        ----------
        params : SimpleSceneModelParameters object
        """
        logp = 0
        for star, star_prior in zip(params.stars, self.star_priors):
            logp += star_prior.evaluate(col=star.col, row=star.row, flux=star.flux)
        if not self.fix_background:
            logp += self.background_prior.evaluate(params.background.flux)
        if not self.fix_focus:
            logp += self.focus_prior.evaluate(params.focus.scale_col,
                                              params.focus.scale_row,
                                              params.focus.rotation_angle)
        return logp

    def _logp_prior(self, params_array):
        """Version of `logp_prior()` that takes an array instead of a
        SimpleSceneModelParameters object.
        """
        params = self.params.from_array(params_array)
        return self.logp_prior(params)

    def fit(self, observed_data, loss_function=PoissonPosterior):
        loss = loss_function(observed_data, self._predict, prior=self._logp_prior)
        fit = loss.fit(x0=self.initial_guesses().to_array(), method='powell')
        result = self.params.from_array(fit.x)
        result.predicted_image = self._predict(*fit.x)
        result.residual_image = observed_data - result.predicted_image
        result.loss_value = fit.fun
        return result

    def plot(self, *params, **kwargs):
        """Plots an image of the model for a given point in the parameter space."""
        img = self.predict(*params)
        plot_image(img,
                   title='Scene Model, Channel: {}'.format(self.prfmodel.channel),
                   extent=(self.prfmodel.column, self.prfmodel.column + self.prfmodel.shape[1],
                           self.prfmodel.row, self.prfmodel.row + self.prfmodel.shape[0]),
                   **kwargs)

    def diagnostics(self, observed_data, *params, **kwargs):
        """Plots an image of the model for a given point in the parameter space."""
        fit = self.fit(observed_data)
        plot_image(observed_data,
                   title='Observed Data, Channel: {}'.format(self.prfmodel.channel),
                   extent=(self.prfmodel.column, self.prfmodel.column + self.prfmodel.shape[1],
                           self.prfmodel.row, self.prfmodel.row + self.prfmodel.shape[0]),
                   **kwargs)
        plot_image(fit.predicted_image,
                   title='Predicted Image, Channel: {}'.format(self.prfmodel.channel),
                   extent=(self.prfmodel.column, self.prfmodel.column + self.prfmodel.shape[1],
                           self.prfmodel.row, self.prfmodel.row + self.prfmodel.shape[0]),
                   **kwargs)
        plot_image(fit.residual_image,
                   title='Residual Image, Channel: {}'.format(self.prfmodel.channel),
                   extent=(self.prfmodel.column, self.prfmodel.column + self.prfmodel.shape[1],
                           self.prfmodel.row, self.prfmodel.row + self.prfmodel.shape[0]),
                   **kwargs)


class PRFPhotometry():
    def __init__(self, model):
        self.model = model
        self.results = []
    
    def fit(self, data):
        self.results = []
        for cadence in tqdm((range(len(data)))):
            self.results.append(model.fit(data[cadence]))
        # Parse results
        self.lightcurves = [self._parse_lightcurve(star_idx)
                            for star_idx in range(len(model.star_priors))]
            
    def _parse_lightcurve(self, star_idx):
        # Create a lightcurve
        flux = []
        for cadence in range(len(self.results)):
            flux.append(self.results[cadence].stars[star_idx].flux)
        return LightCurve(flux=flux, targetid=self.model.star_priors[star_idx].targetid)

    def _parse_background(self):
        # Create a lightcurve
        bgflux = []
        for cadence in range(len(self.results)):
            bgflux.append(self.results[cadence].background.flux)
        return LightCurve(flux=bgflux)
    
    def plot_results(self, star_idx=0):
        fig, ax = pl.subplots(8, sharex=True, figsize=(6, 8))
        x = range(len(self.results))
        ax[0].plot(x, [r.stars[star_idx].flux for r in self.results])
        ax[0].set_ylabel('Flux')
        ax[1].plot(x, [r.stars[star_idx].col for r in self.results])
        ax[1].set_ylabel('Col')
        ax[2].plot(x, [r.stars[star_idx].row for r in self.results])
        ax[2].set_ylabel('Row')
        ax[3].plot(x, [r.background.flux for r in self.results])
        ax[3].set_ylabel('Background')
        ax[4].plot(x, [r.focus.scale_col for r in self.results])
        ax[4].set_ylabel('Focus col')
        ax[5].plot(x, [r.focus.scale_row for r in self.results])
        ax[5].set_ylabel('Focus row')
        ax[6].plot(x, [r.focus.rotation_angle for r in self.results])
        ax[6].set_ylabel('Focus angle')
        ax[7].plot(x, [r.loss_value for r in results])
        ax[7].set_ylabel('Loss')
        return fig


def psf_photometry_demo():
    """Prototype implementation of `KeplerTargetPixelFile.psf_photometry()`"""

    # We will attempt to make a lightcurve for Tabby's star
    tpf = KeplerTargetPixelFile.from_archive(8462852, quarter=16, quality_bitmask='hardest')
    bgflux = np.nanpercentile(tpf.flux[0], 10)
    maxflux = np.nansum(tpf.flux, axis=(1, 2)).max()

    # First, set up a simple scene model with one star and no motion or focus changes
    star_prior = StarPrior(col=GaussianPrior(mean=tpf.column, var=2**2),
                           row=GaussianPrior(mean=tpf.row, var=2**2),
                           flux=UniformPrior(lb=0, ub=maxflux),
                           targetid=tpf.keplerid)
    model = SceneModel(star_priors=[star_prior],
                            background_prior=GaussianPrior(mean=bgflux, var=bgflux),
                            prfmodel=tpf.get_prf_model())

    # Now make the lightcurve by fitting each cadence
    time = tpf.time[1400:1700]  # this cadence range include the deepest dip
    flux = []
    for idx in tqdm(np.arange(1400, 1700)):
        result = model.fit(tpf.flux[idx])
        flux.append(result.stars[0].flux)
    return LightCurve(time, flux)
