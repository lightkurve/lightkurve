from __future__ import division, print_function

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from oktopus import GaussianPrior, UniformPrior, PoissonPosterior
from oktopus.posterior import PoissonPosterior

from .prf import KeplerPRF
from .utils import plot_image


__all__ = ['StarPrior', 'FocusPrior', 'MotionPrior',
           'StarParameters', 'BackgroundParameters', 'FocusParameters',
           'MotionParameters', 'SceneModelParameters',
           'SceneModel', 'PRFPhotometry']


class StarPrior(object):
    """Container class to capture a user's beliefs about a star's position and flux.

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

    def __repr__(self):
        return ('<StarPrior (ID: {}): col={}, row={}, flux={}>'
                ''.format(self.targetid, self.col, self.row, self.flux))

    def evaluate(self, col, row, flux):
        """Evaluate the prior probability of a star of a given flux being at
        a given row and col.
        """
        logp = (self.col.evaluate(col) +
                self.row.evaluate(row) +
                self.flux.evaluate(flux))
        return logp


class FocusPrior():
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
        self.scale_col = scale_col
        self.scale_row = scale_row
        self.rotation_angle = rotation_angle

    def __repr__(self):
        return ('<StarPrior: scale_col={}, scale_row={}, rotation_angle={}>'
                ''.format(self.scale_col, self.scale_row, self.rotation_angle))

    def evaluate(self, scale_col, scale_row, rotation_angle):
        """Returns the prior probability for a gien set of focus parameters."""
        logp = (self.scale_col.evaluate(scale_col) +
                self.scale_row.evaluate(scale_row) +
                self.rotation_angle.evaluate(rotation_angle))
        return logp  


class MotionPrior(object):
    """Container class to capture a user's beliefs about the telescope motion.
    """
    def __init__(self, shift_col=UniformPrior(lb=-0.1, ub=0.1),
                 shift_row=UniformPrior(lb=-0.1, ub=0.1)):
        self.shift_col = shift_col
        self.shift_row = shift_row

    def __repr__(self):
        return ('<MotionPrior: shift_col={}, shift_row={}>'
                ''.format(self.shift_col, self.shift_row))

    def evaluate(self, shift_col, shift_row):
        """Returns the prior probability for a gien set of motion parameters."""
        logp = (self.shift_col.evaluate(shift_col) +
                self.shift_row.evaluate(shift_row))
        return logp


class StarParameters(object):
    """Container class to hold the parameters of a star in a ``SceneModel``.
    """
    def __init__(self, col, row, flux, err_col=None, err_row=None, err_flux=None):
        self.col = col
        self.row = row
        self.flux = flux

    def __repr__(self):
        r = "<StarParameters: col={:.3f}, row={:.3f}, flux={:.3e}>".format(
                    self.col, self.row, self.flux)
        return r


class BackgroundParameters(object):
    """Container class to hold the parameters of the background in a ``SceneModel``.
    """
    def __init__(self, flux=0., err_flux=None, fixed=False):
        self.flux = flux
        self.err_flux = err_flux
        self.fixed = fixed

    def __repr__(self):
        r = "<BackgroundParameters: flux={:.3e}, fixed={}>".format(
                    self.flux, self.fixed)
        return r


class FocusParameters(object):
    """Container class to hold the parameters of the telescope focus in a ``SceneModel``.
    """
    def __init__(self, scale_col=1., scale_row=1., rotation_angle=0., fixed=False):
        self.scale_col = scale_col
        self.scale_row = scale_row
        self.rotation_angle = rotation_angle
        self.fixed = fixed

    def __repr__(self):
        return "<FocusParameters: scale_col={:.3f}, scale_row={:.3f}, rotation_angle={:.3f}, fixed={}>".format(
                    self.scale_col, self.scale_row, self.rotation_angle, self.fixed)


class MotionParameters(object):
    """Container class to hold the parameters of the telescope motion in a ``SceneModel``.
    """
    def __init__(self, shift_col=0., shift_row=0., fixed=False):
        self.shift_col = shift_col
        self.shift_row = shift_row
        self.fixed = fixed

    def __repr__(self):
        return "<MotionParameters: shift_col={:.3f}, shift_row={:.3f}, fixed={}>".format(
                    self.shift_col, self.shift_row, self.fixed)


class SceneModelParameters():
    """Container class to combine all parameters that parameterize a ``SceneModel``.

    Attributes
    ----------
    stars : list of ``StarParameters`` objects
        Parameters related to the stars in the scene.
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
        out = super(SceneModelParameters, self).__repr__() + '\n'
        out += '  Stars:\n'+''.join(['    {}\n'.format(star) for star in self.stars])
        out += '  Background:\n    {}\n'.format(self.background)
        out += '  Focus:\n    {}\n'.format(self.focus)
        out += '  Motion:\n    {}\n'.format(self.motion)
        if 'residual_image' in vars(self):
            out += '  Residual image:\n    {}'.format(self.residual_image[0][0:4])[:-1]
            out +='...\n'
        if 'predicted_image' in vars(self):
            out += '  Predicted image:\n    {}'.format(self.predicted_image[0][0:4])[:-1]
            out +='...\n'
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
        if not self.background.fixed:
            array.append(self.background.flux)
        if not self.focus.fixed:
            array.append(self.focus.scale_col)
            array.append(self.focus.scale_row)
            array.append(self.focus.rotation_angle)
        if not self.motion.fixed:
            array.append(self.motion.shift_col)
            array.append(self.motion.shift_row)
        return np.array(array)

    def from_array(self, array):
        """Inverse of ``to_array()``."""
        next_idx = 0
        stars = []
        for staridx in range(len(self.stars)):
            star = StarParameters(col=array[next_idx],
                                  row=array[next_idx + 1],
                                  flux=array[next_idx + 2])
            stars.append(star)
            next_idx += 3

        if self.background.fixed:
            background = self.background
        else:
            background = BackgroundParameters(flux=array[next_idx])
            next_idx += 1          

        if self.focus.fixed:
            focus = self.focus
        else:
            focus = FocusParameters(scale_col=array[next_idx],
                                    scale_row=array[next_idx + 1],
                                    rotation_angle=array[next_idx + 2])
            next_idx += 3

        if self.motion.fixed:
            motion = self.motion
        else:
            motion = MotionParameters(shift_col=array[next_idx],
                                      shift_row=array[next_idx + 1])
            
        return SceneModelParameters(stars=stars, background=background,
                                    focus=focus, motion=motion)


class SceneModel():
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
    fix_background : bool
        If True, the background parameters will be kept fixed.
    fix_focus : bool
        If True, the telescope focus parameters will be kept fixed.
    fix_motion : bool
        If True, the telescope motion parameters will be kept fixed.
    """
    def __init__(self, star_priors=[],
                 background_prior=GaussianPrior(mean=0, var=100),
                 prfmodel=KeplerPRF(1, shape=(10,10), column=0, row=0),
                 focus_prior=FocusPrior(),
                 motion_prior=MotionPrior(),
                 fix_background=False, fix_focus=True, fix_motion=True):
        self.star_priors = star_priors
        self.background_prior = background_prior
        self.prfmodel = prfmodel
        self.focus_prior = focus_prior
        self.motion_prior = motion_prior
        self.fix_background = fix_background
        self.fix_focus = fix_focus
        self.fix_motion = fix_motion
        self.params = self.initial_guesses()

    def __repr__(self):
        out = super(SceneModel, self).__repr__() + '\n'
        out += '  Star priors:\n'+''.join(['    {}\n'.format(star) for star in self.star_priors])
        out += '  Background prior:\n    {}\n'.format(self.background_prior)
        out += '  Focus prior:\n    {}\n'.format(self.focus_prior)
        out += '  Motion prior:\n    {}\n'.format(self.motion_prior)
        out += '  PRF model:\n    {}\n'.format(self.prfmodel)
        out += '  Fixed parameters:\n    fix_background={}, fix_focus={}, fix_motion={}\n'.format(
                        self.fix_background, self.fix_focus, self.fix_motion)
        return out

    def initial_guesses(self):
        """Returns the prior means which can be used to initialize the model.

        The guesses are obtained by taking the means of the priors.
        """
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
        motion = MotionParameters(shift_col=self.motion_prior.shift_col.mean,
                                  shift_row=self.motion_prior.shift_row.mean,
                                  fixed=self.fix_motion)
        initial_params = SceneModelParameters(stars=initial_star_guesses,
                                              background=background,
                                              focus=focus,
                                              motion=motion)
        return initial_params

    def predict(self, params=None):
        """Returns a synthetic Kepler image given a set of scene parameters.

        Attributes
        ----------
        params : ```SceneModelParameters``` object
            Parameters which define the scene.

        Returns
        -------
        synthetic_image : 2D ndarray
            Predicted image given the parameters.
        """
        if params is None:
            params = self.initial_guesses()
        star_images = []
        for star in params.stars:
            star_images.append(self.prfmodel(flux=star.flux,
                                             center_col=star.col + params.motion.shift_col,
                                             center_row=star.row + params.motion.shift_row,
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
        params = self.params.from_array(params_array)
        return self.predict(params)

    def logp_prior(self, params):
        """Evaluates the prior at a point in the parameter space.

        Attributes
        ----------
        params : SceneModelParameters object
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
        if not self.fix_motion:
            logp += self.motion_prior.evaluate(params.motion.shift_col,
                                               params.motion.shift_row)
        return logp

    def _logp_prior(self, params_array):
        """Wrapper around ``logp_prior()`` which takes an array of shape (n,)
        where n is the number of free parameters.

        Unlike ``predict()`, this function can be called by scipy.optimize.
        """
        params = self.params.from_array(params_array)
        return self.logp_prior(params)

    def fit(self, data, loss_function=PoissonPosterior, method='powell', **kwargs):
        """Fits the scene model to the data.

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
        result : ``SceneParameters`` object
            Fitted parameters plus fitting diagnostics.
        """
        loss = loss_function(data, self._predict, prior=self._logp_prior)
        fit = loss.fit(x0=self.initial_guesses().to_array(), method=method, **kwargs)
        result = self.params.from_array(fit.x)
        result.predicted_image = self._predict(*fit.x)
        result.residual_image = data - result.predicted_image
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

    def diagnostics(self, data, *params, **kwargs):
        """Plots an image of the model for a given point in the parameter space."""
        fit = self.fit(data)
        plot_image(data,
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
    """This class performs PRF Photometry on TPF-like data given a ``SceneModel``.

    This class exists because a ``SceneModel`` object is designed to fit only
    one cadence at a time.  This class makes it easy to fit a large number
    of cadences and obtain the resulting LightCurve.

    Attributes
    ----------
    model : instance of SceneModel
        Model which will be fit to the data
    """
    def __init__(self, model):
        self.model = model
        self.results = []
    
    def run(self, tpf_flux):
        """Fits the scene model to the flux data.

        Parameters
        ----------
        tpf_flux : array-like
            A pixel flux time-series, i.e., the pixel data, e.g,
            KeplerTargetPixelFile.flux, such that (time, row, column) represents
            the shape of ``tpf_flux``.
        """
        self.results = []
        for cadence in tqdm((range(len(tpf_flux)))):
            self.results.append(self.model.fit(tpf_flux[cadence]))
        # Parse results
        self.lightcurves = [self._parse_lightcurve(star_idx)
                            for star_idx in range(len(self.model.star_priors))]
            
    def _parse_lightcurve(self, star_idx):
        # Create a lightcurve
        from . import LightCurve
        flux = []
        for cadence in range(len(self.results)):
            flux.append(self.results[cadence].stars[star_idx].flux)
        return LightCurve(flux=flux, targetid=self.model.star_priors[star_idx].targetid)

    def _parse_background(self):
        # Create a lightcurve
        from . import LightCurve
        bgflux = []
        for cadence in range(len(self.results)):
            bgflux.append(self.results[cadence].background.flux)
        return LightCurve(flux=bgflux)
    
    def plot_results(self, star_idx=0):
        """Plot all the scene model parameters over time."""
        fig, ax = plt.subplots(10, sharex=True, figsize=(6, 10))
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
        return fig


def _example():
    tpf = KeplerTargetPixelFile.from_archive(8462852, quarter=16, quality_bitmask='hardest')
    bgflux = np.nanpercentile(tpf.flux[0], 10)
    maxflux = np.nansum(tpf.flux, axis=(1, 2)).max()

    # First, set up a simple scene model with one star and no motion or focus changes
    col, row = np.nanmedian(tpf.centroids(), axis=1)
    star_prior = StarPrior(col=GaussianPrior(mean=col, var=2**2),
                           row=GaussianPrior(mean=row, var=2**2),
                           flux=UniformPrior(lb=0, ub=maxflux),
                           targetid=tpf.keplerid)
    model = SceneModel(star_priors=[star_prior],
                       background_prior=UniformPrior(lb=0, ub=10),
                       prfmodel=tpf.get_prf_model(),
                       focus_prior=FocusPrior(scale_col=GaussianPrior(mean=1, var=0.0001),
                                              scale_row=GaussianPrior(mean=1, var=0.0001),
                                              rotation_angle=UniformPrior(lb=-3.1415, ub=3.1415)),
                       motion_prior=MotionPrior(shift_col=UniformPrior(lb=-0.1, ub=0.1),
                                                shift_row=UniformPrior(lb=-0.1, ub=0.1)),
                       fix_background=False,
                       fix_focus=False,
                       fix_motion=False)

    pp = PRFPhotometry(model)
    pp.run(tpf.flux[1650:1850])
    pp.plot_results()
