"""Draft implementation of a dream Kepler PSF photometry API."""
import numpy as np
from tqdm import tqdm

from oktopus import GaussianPrior, PoissonPosterior

from .utils import plot_image
from . import LightCurve
from . import KeplerTargetPixelFile


class StarParameters(object):
    """Captures the parameters of a star in a scene model.
    """
    def __init__(self, flux, col, row, err_flux=None, err_col=None, err_row=None):
        self.flux = flux
        self.col = col
        self.row = row
        self.err_flux = err_flux
        self.err_col = err_col
        self.err_row = err_row

    def __repr__(self):
        print('Gorgeous repr.')


class StarPrior(object):
    """Captures the user's beliefs about a single star's position and flux.
    """
    def __init__(self, flux, col, row, err_col=None, err_row=None, err_flux=None, targetid=None):
        self.targetid = targetid
        self.col_prior = GaussianPrior(mean=col, var=err_col**2)
        self.row_prior = GaussianPrior(mean=row, var=err_row**2)
        self.flux_prior = GaussianPrior(mean=flux, var=err_flux**2)

    def evaluate(self, flux, col, row):
        """Evaluate the prior probability of a star of a given flux being at
        a given row and col.
        """
        logp = (self.col_prior.evaluate(col) +
                self.row_prior.evaluate(row) +
                self.flux_prior.evaluate(flux))
        return logp

    def plot(self):
        raise NotImplementedError('Geert was lazy.')


class BackgroundParameters(object):
    """Captures the parameters of the background in a Kepler scene model.
    """
    def __init__(self, flux, err_flux=None):
        self.flux = flux
        self.err_flux = err_flux

    def __repr__(self):
        print('Gorgeous repr.')


class BackgroundPrior(object):
    """Captures the user's beliefs about the background flux.

    Attributes
    ----------
    mean : 2D image or float
        Mean expected background level.
    err_flux : sigma
        Standard deviation around the mean expected background level.
    """
    def __init__(self, flux, err_flux):
        self.flux_prior = GaussianPrior(mean=flux, var=err_flux**2)

    def evaluate(self, flux):
        return self.flux_prior.evaluate(flux)

    def mean(self):
        return BackgroundParameters(flux=self.flux_prior.mean)


class SimpleSceneModelParameters():
    """Parameters that define a single cadence of a TPF image.

    Attributes
    ----------
    stars : list of `StarParameter` objects
        Stars in the scene.
    """
    def __init__(self, stars, background):
        self.stars = stars
        self.background = background

    def to_array(self):
        """Converts the parameters to an array of real elements of size (n,),
        where n is the number of parameters.

        We do this because scipy.optimize can only optimize arrays of real
        numbers.
        """
        array = []
        for star in self.stars:
            array.append(star.flux)
            array.append(star.col)
            array.append(star.row)
        array.append(self.background.flux)
        return np.array(array)

    @staticmethod
    def from_array(array):
        """Converts an array of parameters to a more human-friendly
        `SceneModelParameters class`.
        """
        stars = []
        n_stars = int((len(array) - 1) / 3)
        for staridx in range(n_stars):
            star = StarParameters(flux=array[staridx * 3],
                                  col=array[staridx * 3 + 1],
                                  row=array[staridx * 3 + 2])
            stars.append(star)
        background = BackgroundParameters(flux=array[-1])
        return SimpleSceneModelParameters(stars=stars, background=background)


class SimpleSceneModel():
    """A model which describes a single-cadence Kepler image.

    Attributes
    ----------
    star_priors : list of `StarPrior` objects.
        List of stars believed to be in the image.
    background_prior : BackgroundPrior object.
        Beliefs about the per-pixel background flux.
    prfmodel : KeplerPRF object.
    """
    def __init__(self, star_priors, background_prior, prfmodel):
        self.star_priors = star_priors
        self.background_prior = background_prior
        self.prfmodel = prfmodel

    def initial_guesses(self):
        """Returns the prior means."""
        initial_star_guesses = []
        for star in self.star_priors:
            initial_star_guesses.append(StarParameters(flux=star.flux_prior.mean,
                                                       col=star.col_prior.mean,
                                                       row=star.row_prior.mean))
        background = BackgroundParameters(flux=self.background_prior.flux_prior.mean)
        initial_params = SimpleSceneModelParameters(stars=initial_star_guesses,
                                                    background=background)
        assert self.logp_prior(initial_params) == 0
        return initial_params

    def plot(self, params=None):
        """Plots an image of the model for a given point in the parameter space."""
        if params is None:
            params = self.initial_guesses()
        img = self.predict(params)
        plot_image(img)

    def predict(self, params=None):
        """Produces a synthetic Kepler 2D image given a set of scene parameters.

        Attributes
        ----------
        params : SimpleSceneModelParameters object
        """
        if params is None:
            params = self.initial_guesses()
        star_images = []
        for star in params.stars:
            star_images.append(self.prfmodel(star.flux, star.col, star.row))
        synthetic_image = np.sum(star_images, axis=0) + params.background.flux
        return synthetic_image

    def _predict(self, *params_array):
        """Version of `predict()` that takes an array instead of a
        SimpleSceneModelParameters object.
        """
        params = SimpleSceneModelParameters.from_array(params_array)
        return self.predict(params)

    def logp_prior(self, params):
        """Evaluates the prior at a point in the parameter space.

        Attributes
        ----------
        params : SimpleSceneModelParameters object
        """
        logp = 0
        for star, star_prior in zip(params.stars, self.star_priors):
            logp += star_prior.evaluate(flux=star.flux, col=star.col, row=star.row)
        logp += self.background_prior.evaluate(flux=params.background.flux)
        return logp

    def _logp_prior(self, params_array):
        """Version of `logp_prior()` that takes an array instead of a
        SimpleSceneModelParameters object.
        """
        params = SimpleSceneModelParameters.from_array(params_array)
        return self.logp_prior(params)

    def fit(self, observed_data, loss_function=PoissonPosterior):
        loss = loss_function(observed_data, self._predict, prior=self._logp_prior)
        result = loss.fit(x0=self.initial_guesses().to_array(), method='powell')
        opt_params = result.x
        residuals = observed_data - self._predict(*opt_params)
        self.loss_value = result.fun
        self.opt_params = opt_params
        self.residuals = residuals
        return SimpleSceneModelParameters.from_array(opt_params)


def psf_photometry_demo():

    # We will attempt to make a lightcurve for Tabby's star
    tpf = KeplerTargetPixelFile.from_archive(8462852, quarter=16, quality_bitmask='hardest')

    # First, set up a simple scene model with one star and no motion or focus changes
    star_prior = StarPrior(flux=np.nansum(tpf.flux[0]), col=tpf.column, row=tpf.row,
                           err_flux=1e5, err_col=2, err_row=2,
                           targetid=tpf.keplerid)
    background_prior = BackgroundPrior(flux=np.nanpercentile(tpf.flux[0], 10), err_flux=100.0)
    scenemodel = SimpleSceneModel(star_priors=[star_prior],
                                  background_prior=background_prior,
                                  prfmodel=tpf.get_prf_model())

    # Now make the lightcurve by fitting each cadence
    time = tpf.time[1400:1700]
    flux = []
    for idx in tqdm(range(len(time))):
        result = scenemodel.fit(tpf.flux[idx])
        flux.append(result.stars[0].flux)
    return LightCurve(time, flux)

"""
# Possibe changes to TPF planned:

class KeplerTargetPixelFile():

    def to_lightcurve(method='aperture'):
        ...

    def aperture_photometry(self):
        ...

    def get_scenemodel(self):
        return ...

    def psf_photometry(self, sourcelist, return_diagnostics=False):
        star_priors = tpf.get_starpriors(sourcelist)
        BackgroudPrior(self.data.median(), self.data.std())
        scenemodel = self.get_scenemodel(star_priors)
        for cadence in tpf.data:
            result = scenemodel.fit(tpf.data)
            results.append(result)
        for idx in range(len(r.stars)):
            LightCurve(flux=[r.stars[0].flux for r in results])
        return LightCurveCollection([lc1, lc2, lc3])
"""
