"""Draft implementation of a dream Kepler PSF photometry API."""
import numpy as np

from oktopus import UniformPrior, GaussianPrior, PoissonPosterior

from .utils import plot_image
from .prf import SimpleKeplerPRF


class StarParameters(object):
    """Captures the parameters of a star in a model.
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
    """Captures user's beliefs about a star's position and flux.
    """
    def __init__(self, flux, col, row, err_col=None, err_row=None, err_flux=None, targetid=None):
        self.targetid = targetid
        self.col_prior = GaussianPrior(mean=col, var=err_col**2)
        self.row_prior = GaussianPrior(mean=row, var=err_row**2)
        self.flux_prior = GaussianPrior(mean=flux, var=err_flux**2)

    def evaluate(self, flux, col, row):
        """Evaluate the prior probability of a star of a given flux being at
        a given row and col."""
        logp = (self.col_prior.evaluate(col) +
                self.row_prior.evaluate(row) +
                self.flux_prior.evaluate(flux))
        return logp

    def plot(self):
        raise NotImplementedError('Geert was lazy.')


class BackgroundParameters(object):
    """Captures the parameters of the background in a model.
    """
    def __init__(self, flux, err_flux=None):
        self.flux = flux
        self.err_flux = err_flux

    def __repr__(self):
        print('Gorgeous repr.')


class BackgroundPrior(object):
    """Captures user's beliefs about the background level.

    Attributes
    ----------
    mean : 2D image or float
    """
    def __init__(self, flux, err_flux):
        self.flux_prior = GaussianPrior(mean=flux, var=err_flux**2)

    def evaluate(self, flux):
        return self.flux_prior.evaluate(flux)

    def mean(self):
        return BackgroundParameters(flux=self.flux_prior.mean)


class SceneModelParameters():
    """Parameters that define a single cadence of a TPF image.

    Attributes
    ----------
    stars : list of `StarParameter` objects
        Stars in the scene.
    """
    def __init__(self, stars, background):
        self.stars = stars
        self.background = background

    def to_tuple(self):
        """Convert to a tuple of numbers, which is what oktopus needs."""
        result = []
        for star in self.stars:
            result.append(star.flux)
            result.append(star.col)
            result.append(star.row)
        result.append(background.flux)
        return tuple(result)


class SimpleSceneModel():
    """A model which describes a single Kepler image.

    Attributes
    ----------
    star_priors : list of `StarPrior` objects
    """
    def __init__(self, star_priors, background_prior, prfmodel):
        self.star_priors = star_priors
        self.background_prior = background_prior
        self.prfmodel = prfmodel

    def initial_guesses(self):
        # Infer initial parameter guesses from the prior means
        initial_star_guesses = []
        for star in self.star_priors:
            initial_star_guesses.append(StarParameters(flux=star.flux_prior.mean,
                                                       col=star.col_prior.mean,
                                                       row=star.row_prior.mean))
        initial_params = SceneModelParameters(stars=initial_star_guesses,
                                              background=BackgroundParameters(flux=self.background_prior.flux_prior.mean))
        return initial_params

    def plot(self, params=None):
        if params is None:
            params = self.initial_guesses()
        img = self.predict(params)
        plot_image(img)

    def predict(self, params=None):
        """Produces a synthetic Kepler 2D image given a set of scene parameters."""
        # put a star at position col + delta_col
        if params is None:
            params = self.initial_guesses()
        star_images = []
        for star in params.stars:
            star_images.append(self.prfmodel(star.flux, star.col, star.row))
        synthetic_image = np.sum(star_images, axis=0) + params.background.flux
        return synthetic_image

    def evaluate_prior(self, params):
        logp = 0
        for star, star_prior in zip(params.stars, self.star_priors):
            logp += star_prior.evaluate(flux=star.flux, col=star.col, row=star.row)
        logp += self.background_prior.evaluate(flux=params.background.flux)
        return logp

    def evaluate_likelihood(self, params, data):
        # Poisson likelihood
        return np.nansum(self.predict(params) - data * np.log(self.predict(params)))

    def evaluate_posterior(self, params, data):
        return self.evaluate_likelihood(params, data) + self.evaluate_prior(params)

    def fit(self, observed_data, loss_function=PoissonPosterior):
        from scipy.optimize import minimize
        self.opt_result = minimize(self.evaluate_posterior, x0=self.initial_guesses().to_tuple())

        pass
        loss_function
        loss = self.loss_function(tpf_flux[t], self.predict,
                                      prior=self.prior)
        """
        self.fitted_params = self.params
        score = self.predict(self.fitted_params) - observed_data
        whatevz = optimize(score)
        self.fitted_params.meta['scipy'] = whatevz
        return self.fitted_params
        """

def scenemodelparameters_to_listofparameters()

"""

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


def do():
    stars = [Star(flux=100, col=10, row=20, err_flux=0.1, err_col=1, err_row=1, targetid="christina")]
    prfmodel = SimpleKeplerPRF(channel=44, shape=(10, 10), column=10, row=20)
    background = Background(bgflux=10, sigma_bgflux=1.0)
    scenemodel = SimpleSceneModel(stars=stars, prfmodel=prfmodel, background=background)
    img = scenemodel.predict()
    return img


if __name__ == '__main__':
    #tpf = KeplerTargetPixelFile()
    #tpf.psf_photometry(sourcelist='kic')
    pass
