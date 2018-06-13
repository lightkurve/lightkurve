import numpy as np

from oktopus import UniformPrior, GaussianPrior, PoissonPosterior

from .utils import plot_image
from .prf import SimpleKeplerPRF


class Star(object):
    """Holds the information on a star being fitted during PSF photometry."""
    def __init__(self, flux, col, row, err_col=None, err_row=None, err_flux=None, targetid=None):
        self.flux = flux
        self.col = col
        self.row = row
        self.targetid = targetid
        self.err_col = err_col
        self.err_row = err_row
        self.err_flux = err_flux
        self.col_prior = GaussianPrior(mean=self.col, var=self.err_col**2)
        self.row_prior = GaussianPrior(mean=self.row, var=self.err_row**2)
        self.flux_prior = GaussianPrior(mean=self.flux, var=self.err_flux**2)

    def evaluate(self, flux, col, row):
        """Evaluate the prior probability of a star of a given flux being at
        a given row and col."""
        logp = (self.col_prior.evaluate(col) +
                self.row_prior.evaluate(row) +
                self.flux_prior.evaluate(flux))
        return logp

    def plot(self):
        raise NotImplementedError('Geert was lazy.')


class Background(object):
    """
    Attributes
    ----------
    mean : 2D image or float
    """
    def __init__(self, bgflux, sigma_bgflux):
        self.bgflux = bgflux
        self.sigma_bgflux = sigma_bgflux
        self.prior = GaussianPrior(mean=self.bgflux, var=self.sigma_bgflux**2)

    def evaluate(self, bgflux):
        return self.prior.evaluate(bgflux)


class SceneModelParameters():
    """Parameters that define a single cadence of a TPF image.

    Attributes
    ----------
    stars : list of `Star` objects
        Stars in the scene.
    """
    def __init__(self, stars, background=None, meta=None):
        self.stars = stars
        self.background = background
        self.meta = meta  # intended to hold scipy fitting diagnostics (aka garbage)

    def evaluate(self, params):
        """Returns probability of a new set of params given the current priors."""
        logp = 0
        for old_star, new_star in zip(self.stars, params.stars):
            logp += old_star.evaluate(new_star.flux, new_star.col, new_star.row)
        logp += self.background.evaluate(params.background.bgflux)
        return logp


class SimpleSceneModel():
    """A model which describes a single Kepler image.

    Attributes
    ----------
    stars : list of `Star` objects
    """
    def __init__(self, stars, prfmodel, background):
        self.initial_params = SceneModelParameters(stars, background)
        self.prfmodel = prfmodel

    def plot(self, params=None):
        if params is None:
            params = self.initial_params
        img = self.predict(params)
        plot_image(img)

    def predict(self, params=None):
        """Produces a synthetic Kepler 2D image given a set of scene parameters."""
        # put a star at position col + delta_col
        if params is None:
            params = self.initial_params
        star_images = []
        for star in params.stars:
            star_images.append(self.prfmodel(star.flux, star.col, star.row))
        synthetic_image = np.sum(star_images, axis=0) + params.background.bgflux
        return synthetic_image

    def fit(self, observed_data, loss_function=PoissonPosterior):
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
