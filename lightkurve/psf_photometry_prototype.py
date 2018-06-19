"""Draft implementation of a dream Kepler PSF photometry API."""
import numpy as np
from tqdm import tqdm

from oktopus import GaussianPrior, UniformPrior, PoissonPosterior

from .utils import plot_image
from . import LightCurve
from . import KeplerTargetPixelFile


class StarParameters(object):
    """Captures the parameters of a star in a scene model.
    """
    def __init__(self, col, row, flux, err_col=None, err_row=None, err_flux=None):
        self.col = col
        self.row = row
        self.flux = flux
        self.dict = {0:self.flux, 1:self.col, 2:self.row}
        self.label_dict = {0:'flux', 1:'col', 2:'row'}

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < 3:
            self.n += 1
            return (self.label_dict[self.n-1], self.dict[self.n-1])
        else:
            raise StopIteration

    next = __next__

    def __repr__(self):
        return ('StarParameters: col: {}, row: {}, flux: {}'
                ''.format(self.col, self.row, self.flux))


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
        self.dict = {0:self.flux, 1:self.col, 2:self.row}
        self.label_dict = {0:'flux', 1:'col', 2:'row'}

    def __repr__(self):
        return ('StarPrior (ID: {}):\n \tcol: {}\n \trow: {}\n \tflux: {}\n'
                ''.format(self.targetid, self.col, self.row, self.flux))

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < 3:
            self.n += 1
            return (self.label_dict[self.n-1], self.dict[self.n-1])
        else:
            raise StopIteration

    next = __next__

    def evaluate(self, col, row, flux):
        """Evaluate the prior probability of a star of a given flux being at
        a given row and col.
        """
        logp = (self.col.evaluate(col) +
                self.row.evaluate(row) +
                self.flux.evaluate(flux))
        return logp


class BackgroundParameters(object):
    """Captures the parameters of the background in a Kepler scene model.
    """
    def __init__(self, flux, err_flux=None):
        self.flux = flux
        self.err_flux = err_flux

    def __repr__(self):
        return ('BackgroundParameters: flux: {}, err_flux: {}'
                ''.format(self.flux, self.err_flux))


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

    def __repr__(self):
        output = 'SimpleSceneModelParameters\n'
        output += '\t Stars:\n'+''.join(['\t\t{}\n'.format(star) for star in self.stars])
        output += '\t Background:\n\t\t{}\n'.format(self.background)
        if 'residual_image' in vars(self):
            output += '\t Residual Image: \n\t\t {}'.format(self.residual_image[0][0:4])[:-1]
            output +='...\n\t\t\t...\n'
        if 'predicted_image' in vars(self):
            output += '\t Predicted Image: \n\t\t {}'.format(self.predicted_image[0][0:4])[:-1]
            output +='...\n\t\t\t...\n'

        return output

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
        if isinstance(array, tuple):
            array = np.asarray(array).ravel()
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

    def __repr__(self):
        s = '\t Stars Priors:\n'+''.join(['\t\t{}\n'.format(star) for star in self.star_priors])
        b = '\t Background Prior:\n\t\t{}\n'.format(self.background_prior)
        return 'SimpleSceneModel\n'+s+b


    def initial_guesses(self):
        """Returns the prior means."""
        initial_star_guesses = []
        for star in self.star_priors:
            initial_star_guesses.append(StarParameters(col=star.col.mean,
                                                       row=star.row.mean,
                                                       flux=star.flux.mean))
        background = BackgroundParameters(flux=self.background_prior.mean)
        initial_params = SimpleSceneModelParameters(stars=initial_star_guesses,
                                                    background=background)
        return initial_params

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
            logp += star_prior.evaluate(col=star.col, row=star.row, flux=star.flux)
        logp += self.background_prior.evaluate(params.background.flux)
        return logp

    def _logp_prior(self, params_array):
        """Version of `logp_prior()` that takes an array instead of a
        SimpleSceneModelParameters object.
        """
        params = SimpleSceneModelParameters.from_array(params_array)
        return self.logp_prior(params)

    def fit(self, observed_data, loss_function=PoissonPosterior):
        loss = loss_function(observed_data, self._predict, prior=self._logp_prior)
        fit = loss.fit(x0=self.initial_guesses().to_array(), method='powell')
        result = SimpleSceneModelParameters.from_array(fit.x)
        result.predicted_image = self._predict(*fit.x)
        result.residual_image = observed_data - result.predicted_image
        result.loss_value = fit.fun
        return result


def psf_photometry_demo():
    """Prototype implementation of `KeplerTargetPixelFile.psf_photometry()`"""

    # We will attempt to make a lightcurve for Tabby's star
    tpf = KeplerTargetPixelFile.from_archive(8462852, quarter=16, quality_bitmask='hardest')
    bgflux = np.nanpercentile(tpf.flux[0], 10)
    maxflux = np.nansum(tpf.flux, axis=(1, 2)).max()

    # First, set up a simple scene model with one star and no motion or focus changes
    col, row = np.nanmedian(tpf.centroids(), axis=1)
    star_prior = StarPrior(col=UniformPrior(lb=col-4, ub=col+4),
                           row=UniformPrior(lb=row-4, ub=row+4),
                           flux=UniformPrior(lb=0, ub=maxflux),
                           targetid=tpf.keplerid)
    model = SimpleSceneModel(star_priors=[star_prior],
                             background_prior=UniformPrior(lb=0, ub=10),
                             prfmodel=tpf.get_prf_model())

    # Now make the lightcurve by fitting each cadence
    time = tpf.time[1400:1700]  # this cadence range include the deepest dip
    flux, cols, rows, bkgs = [], [], [], []
    for idx in tqdm(np.arange(1400, 1700)):
        result = model.fit(tpf.flux[idx])
        flux.append(result.stars[0].flux)
        cols.append(result.stars[0].col)
        rows.append(result.stars[0].row)
        bkgs.append(result.background.flux)
    return LightCurve(time, flux)
