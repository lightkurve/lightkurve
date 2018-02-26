from __future__ import division, print_function

import math
import sys

from astropy.io import fits as pyfits
import numpy as np
import scipy
import tqdm
from oktopus.posterior import PoissonPosterior

from .utils import channel_to_module_output, plot_image

# This is a workaround to get the number of arguments of
# a given function.
# In Python 2, this works by using getargspec.
# Note that `self` is accounted as an argument,
# which is unwanted, hence the subtraction by 1.
# On the other hand, Python 3 handles that trivially with the
# signature function.
if sys.version_info[0] == 2:
    from inspect import getargspec
    def _get_number_of_arguments(func):
        list_of_args = getargspec(func).args
        if 'self' in list_of_args:
            return len(list_of_args) - 1
        else:
            return len(list_of_args)
else:
    from inspect import signature
    def _get_number_of_arguments(func):
        return len(signature(func).parameters)


__all__ = ['KeplerPRF', 'PRFPhotometry', 'SceneModel', 'SimpleKeplerPRF',
           'get_initial_guesses']

class KeplerPRF(object):
    """
    Kepler's Pixel Response Function as designed by [1]_.

    This class provides the necessary interface to load Kepler PRF
    calibration files and to create a model that can be fit as a function
    of flux, center positions, width, and rotation angle.

    Attributes
    ----------
    channel : int
        KeplerTargetPixelFile.channel
    shape : (int, int)
        KeplerTargetPixelFile.shape[1:]
    column : int
        KeplerTargetPixelFile.column
    row : int
        KeplerTargetPixelFile.row

    Examples
    --------
    Objects from the KeplerPRF class are defined by a channel number, a pair of
    dimensions (the size of the image), and a reference coordinate (bottom left
    corner). In this example, we create a KeplerPRF object located at channel
    #44 with dimension equals 10 x 10, reference row and column coordinate
    equals (5, 5). After the object has been created, we may translate it to a
    given center coordinate. Additionally, we can specify total flux, pixel
    scales, and rotation around the object's center.

    >>> import math
    >>> import matplotlib.pyplot as plt
    >>> from lightkurve import KeplerPRF
    >>> kepprf = KeplerPRF(channel=44, shape=(10, 10), column=5, row=5) # doctest: +SKIP
    Downloading http://archive.stsci.edu/missions/kepler/fpc/prf
    /extracted/kplr13.4_2011265_prf.fits [Done]
    >>> prf = kepprf(flux=1000, center_col=10, center_row=10,
    ...              scale_row=0.7, scale_col=0.7, rotation_angle=math.pi/2) # doctest: +SKIP
    >>> plt.imshow(prf, origin='lower') # doctest: +SKIP

    References
    ----------
    .. [1] S. T. Bryson. The Kepler Pixel Response Function, 2010.
           <https://arxiv.org/abs/1001.0331>.
    """

    def __init__(self, channel, shape, column, row):
        self.channel = channel
        self.shape = shape
        self.column = column
        self.row = row
        self.col_coord, self.row_coord, self.interpolate = self._prepare_prf()

    def __call__(self, flux, center_col, center_row, scale_col, scale_row,
                 rotation_angle):
        return self.evaluate(flux, center_col, center_row,
                             scale_col, scale_row, rotation_angle)

    def evaluate(self, flux, center_col, center_row, scale_col, scale_row,
                 rotation_angle):
        """
        Interpolates the PRF model onto detector coordinates.

        Parameters
        ----------
        flux : float
            Total integrated flux of the PRF
        center_col, center_row : float
            Column and row coordinates of the center
        scale_col, scale_row : float
            Pixel scale in the column and row directions
        rotation_angle : float
            Rotation angle in radians

        Returns
        -------
        prf_model : 2D array
            Two dimensional array representing the PRF values parametrized
            by flux, centroids, widths, and rotation.
        """
        cosa = math.cos(rotation_angle)
        sina = math.sin(rotation_angle)

        delta_col = self.col_coord - center_col
        delta_row = self.row_coord - center_row
        delta_col, delta_row = np.meshgrid(delta_col, delta_row)

        rot_row = delta_row * cosa - delta_col * sina
        rot_col = delta_row * sina + delta_col * cosa

        self.prf_model = flux * self.interpolate(rot_row.flatten() * scale_row,
                                                 rot_col.flatten() * scale_col, grid=False).reshape(self.shape)
        return self.prf_model

    def _read_prf_calibration_file(self, path, ext):
        prf_cal_file = pyfits.open(path)
        data = prf_cal_file[ext].data
        # looks like these data below are the same for all prf calibration files
        crval1p = prf_cal_file[ext].header['CRVAL1P']
        crval2p = prf_cal_file[ext].header['CRVAL2P']
        cdelt1p = prf_cal_file[ext].header['CDELT1P']
        cdelt2p = prf_cal_file[ext].header['CDELT2P']
        prf_cal_file.close()

        return data, crval1p, crval2p, cdelt1p, cdelt2p

    def _prepare_prf(self):
        n_hdu = 5
        min_prf_weight = 1e-6
        module, output = channel_to_module_output(self.channel)
        # determine suitable PRF calibration file
        if module < 10:
            prefix = 'kplr0'
        else:
            prefix = 'kplr'
        prfs_url_path = "http://archive.stsci.edu/missions/kepler/fpc/prf/extracted/"
        prffile = prfs_url_path + prefix + str(module) + '.' + str(output) + '_2011265_prf.fits'

        # read PRF images
        prfn = [0] * n_hdu
        crval1p = np.zeros(n_hdu, dtype='float32')
        crval2p = np.zeros(n_hdu, dtype='float32')
        cdelt1p = np.zeros(n_hdu, dtype='float32')
        cdelt2p = np.zeros(n_hdu, dtype='float32')

        for i in range(n_hdu):
            prfn[i], crval1p[i], crval2p[i], cdelt1p[i], cdelt2p[i] = self._read_prf_calibration_file(prffile, i+1)

        prfn = np.array(prfn)
        PRFcol = np.arange(0.5, np.shape(prfn[0])[1] + 0.5)
        PRFrow = np.arange(0.5, np.shape(prfn[0])[0] + 0.5)
        PRFcol = (PRFcol - np.size(PRFcol) / 2) * cdelt1p[0]
        PRFrow = (PRFrow - np.size(PRFrow) / 2) * cdelt2p[0]

        # interpolate the calibrated PRF shape to the target position
        rowdim, coldim = self.shape[0], self.shape[1]
        prf = np.zeros(np.shape(prfn[0]), dtype='float32')
        prf_weight = np.zeros(n_hdu, dtype='float32')
        ref_column = self.column + .5 * coldim
        ref_row = self.row + .5 * rowdim

        for i in range(n_hdu):
            prf_weight[i] = math.sqrt((ref_column - crval1p[i]) ** 2
                                     + (ref_row - crval2p[i]) ** 2)
            if prf_weight[i] < min_prf_weight:
                prf_weight[i] = min_prf_weight
            prf += prfn[i] / prf_weight[i]

        prf /= (np.nansum(prf) * cdelt1p[0] * cdelt2p[0])

        # location of the data image centered on the PRF image (in PRF pixel units)
        col_coord = np.arange(self.column + .5, self.column + coldim + .5)
        row_coord = np.arange(self.row + .5, self.row + rowdim + .5)
        # x-axis correspond to row-axis in scipy.RectBivariate
        # not to be confused with our convention, in which the
        # x-axis correspond to the column-axis
        interpolate = scipy.interpolate.RectBivariateSpline(PRFrow, PRFcol, prf)

        return col_coord, row_coord, interpolate

    def plot(self, *params, **kwargs):
        pflux = self.evaluate(*params)
        plot_image(pflux, title='Kepler PRF Model, Channel: {}'.format(self.channel),
                   extent=(self.column, self.column + self.shape[1],
                           self.row, self.row + self.shape[0]), **kwargs)

class PRFPhotometry(object):
    """
    This class performs PRF Photometry on TPF-like files.

    Attributes
    ----------
    scene_model : instance of SceneModel
        Model which will be fit to the data
    priors : instance of oktopus.JointPrior
        Priors on the parameters that will be estimated
    loss_function : subclass of oktopus.LossFunction
        Noise distribution associated with each random measurement

    Examples
    --------
    >>> from lightkurve import KeplerTargetPixelFile, SimpleKeplerPRF, SceneModel, PRFPhotometry
    >>> from oktopus import UniformPrior
    >>> tpf = KeplerTargetPixelFile("https://archive.stsci.edu/missions/kepler/"
    ...                             "target_pixel_files/0084/008462852/"
    ...                             "kplr008462852-2013098041711_lpd-targ.fits.gz") # doctest: +SKIP
    Downloading https://archive.stsci.edu/missions/kepler/target_pixel_files
    /0084/008462852/kplr008462852-2013098041711_lpd-targ.fits.gz [Done]
    >>> prf = tpf.get_prf_model() # doctest: +SKIP
    Downloading http://archive.stsci.edu/missions/kepler/fpc/prf
    /extracted/kplr16.4_2011265_prf.fits [Done]
    >>> scene = SceneModel(prfs=prf) # doctest: +SKIP
    >>> prior = UniformPrior(lb=[1.2e5, 230., 128.,1e2], ub=[3.4e5, 235., 133., 1e3]) # doctest: +SKIP
    >>> phot = PRFPhotometry(scene, prior) # doctest: +SKIP
    >>> results = phot.fit(tpf.flux) # doctest: +SKIP
    >>> flux_fit = results[:, 0] # doctest: +SKIP
    >>> x_fit = results[:, 1] # doctest: +SKIP
    >>> y_fit = results[:, 2] # doctest: +SKIP
    >>> bkg_fit = results[:, 3] # doctest: +SKIP
    """

    def __init__(self, scene_model, prior, loss_function=PoissonPosterior,
                 **kwargs):
        self.scene_model = scene_model
        self.prior = prior
        self.loss_function = loss_function
        self.loss_kwargs = kwargs
        self.opt_params = np.array([])
        self.residuals = np.array([])
        self.loss_value = np.array([])
        self.uncertainties = np.array([])

    def fit(self, tpf_flux, x0=None, cadences='all', method='powell',
            **kwargs):
        """
        Fits the scene model to the given data in ``tpf_flux``.

        Parameters
        ----------
        tpf_flux : array-like
            A pixel flux time-series, i.e., the pixel data, e.g,
            KeplerTargetPixelFile.flux, such that (time, row, column) represents
            the shape of ``tpf_flux``.
        x0 : array-like or None
            Initial guesses on the parameters. The default is to use the mean
            of the prior distribution.
        cadences : array-like of ints or str
            A list or array that contains the cadences which will be fitted.
            Default is to fit all cadences.
        kwargs : dict
            Dictionary of additional parameters to be passed to
            `scipy.optimize.minimize`.

        Returns
        -------
        opt_params : array-like
            Matrix with the optimized parameter values. The i-th line contain
            the best parameter values at the i-th cadence. The order of the parameters
            in every line follows the order of the ``scene_model``.
        """
        self.opt_params = np.array([])
        self.residuals = np.array([])
        self.loss_value = np.array([])
        self.uncertainties = np.array([])

        if x0 is None:
            x0 = self.prior.mean

        if cadences == 'all':
            cadences = range(tpf_flux.shape[0])

        for t in tqdm.tqdm(cadences):
            loss = self.loss_function(tpf_flux[t], self.scene_model,
                                      prior=self.prior, **self.loss_kwargs)
            result = loss.fit(x0=x0, method='powell', **kwargs)
            opt_params = result.x
            residuals = tpf_flux[t] - self.scene_model(*opt_params)
            self.loss_value = np.append(self.loss_value, result.fun)
            self.opt_params = np.append(self.opt_params, opt_params)
            self.residuals = np.append(self.residuals, residuals)
        self.opt_params = self.opt_params.reshape((tpf_flux.shape[0], len(x0)))
        self.residuals = self.residuals.reshape(tpf_flux.shape)

        return self.opt_params

    def get_residuals(self):
        return self.residuals


class SceneModel(object):
    """
    This class builds a generic model for a scene.

    Attributes
    ----------
    prfs : list of callables
        A list of prfs
    bkg_model : callable
        A function that models the background variation.
        Default is a constant background
    """

    def __init__(self, prfs, bkg_model=lambda bkg: np.array([bkg])):
        self.prfs = np.asarray([prfs]).reshape(-1)
        self.bkg_model = bkg_model
        self._prepare_scene_model()

    def __call__(self, *params):
        return self.evaluate(*params)

    def _prepare_scene_model(self):
        self.n_models = len(self.prfs)
        self.bkg_order = _get_number_of_arguments(self.bkg_model)

        model_orders = [0]
        for i in range(self.n_models):
            model_orders.append(_get_number_of_arguments(self.prfs[i].evaluate))
        self.n_params = np.cumsum(model_orders)

    def evaluate(self, *params):
        """
        Parameters
        ----------
        flux : scalar or array-like
            Total integrated flux of the PRF model
        center_col, center_row : scalar or array-like
            Column and row coordinates of the center
        scale_col, scale_row : scalar or array-like
            Pixel scale in the column and row directions
        rotation_angle : float
            Rotation angle in radians
        bkg_params : scalar or array-like
            Parameters for the background model
        """
        self.mm = []
        for i in range(self.n_models):
            self.mm.append(self.prfs[i](*params[self.n_params[i]:self.n_params[i+1]]))
        self.scene_model = np.sum(self.mm, axis=0) + self.bkg_model(*params[-self.bkg_order:])
        return self.scene_model

    def gradient(self, *params):
        grad = []
        for i in range(self.n_models):
            grad.append(self.prfs[i].gradient(*params[self.n_params[i]:self.n_params[i+1]]))
        grad.append(self.bkg_model.gradient(*params[-self.bkg_order:]))
        grad = sum(grad, [])
        return grad

    def plot(self, *params, **kwargs):
        pflux = self.evaluate(*params)
        plot_image(pflux, title='Scene Model, Channel: {}'.format(self.prfs[0].channel),
                   extent=(self.prfs[0].column, self.prfs[0].column + self.prfs[0].shape[1],
                           self.prfs[0].row, self.prfs[0].row + self.prfs[0].shape[0]), **kwargs)


class SimpleKeplerPRF(KeplerPRF):
    """
    Simple model of KeplerPRF.

    This class provides identical functionality as in KeplerPRF, except that
    it is parametrized only by flux and center positions. The width scales
    and angle are fixed to 1.0 and 0, respectivelly.
    """

    def __call__(self, flux, center_col, center_row):
        return self.evaluate(flux, center_col, center_row)

    def evaluate(self, flux, center_col, center_row):
        """
        Interpolates the PRF model onto detector coordinates.

        Parameters
        ----------
        flux : float
            Total integrated flux of the PRF
        center_col, center_row : float
            Column and row coordinates of the center

        Returns
        -------
        prf_model : 2D array
            Two dimensional array representing the PRF values parametrized
            by flux and centroids.
        """
        delta_col = self.col_coord - center_col
        delta_row = self.row_coord - center_row
        self.prf_model = flux * self.interpolate(delta_row, delta_col)

        return self.prf_model

    def gradient(self, flux, center_col, center_row):
        """
        This function returns the gradient of the SimpleKeplerPRF model with
        respect to flux, center_col, and center_row.

        Parameters
        ----------
        flux : float
            Total integrated flux of the PRF
        center_col, center_row : float
            Column and row coordinates of the center

        Returns
        -------
        grad_prf : list
            Returns a list of arrays where the elements are the derivative
            of the KeplerPRF model with respect to flux, center_col, and
            center_row, respectively.
        """
        delta_col = self.col_coord - center_col
        delta_row = self.row_coord - center_row

        deriv_flux = self.interpolate(delta_row, delta_col)
        deriv_center_col = - flux * self.interpolate(delta_row, delta_col, dy=1)
        deriv_center_row = - flux * self.interpolate(delta_row, delta_col, dx=1)

        return [deriv_flux, deriv_center_col, deriv_center_row]


def get_initial_guesses(data, ref_col, ref_row):
    """
    Compute the initial guesses for total flux, centers position, and PSF
    width using the sample moments of the data.

    Parameters
    ----------
    data : 2D array-like
        Image data
    ref_col, ref_row : scalars
        Reference column and row (coordinates of the bottom left corner)

    Return
    ------
    flux0, col0, row0, sigma0: floats
        Inital guesses for flux, center position, and width
    """

    flux0 = np.nansum(data)
    yy, xx = np.indices(data.shape) + 0.5
    yy = ref_row + yy
    xx = ref_col + xx
    col0 = np.nansum(xx * data) / flux0
    row0 = np.nansum(yy * data) / flux0
    marg_col = data[:, int(np.round(col0 - ref_col))]
    marg_row = data[int(np.round(row0 - ref_row)), :]
    sigma_y = math.sqrt(np.abs((np.arange(marg_row.size) - row0) ** 2 * marg_row).sum() / marg_row.sum())
    sigma_x = math.sqrt(np.abs((np.arange(marg_col.size) - col0) ** 2 * marg_col).sum() / marg_col.sum())
    sigma0 = math.sqrt((sigma_x**2 + sigma_y**2)/2.0)

    return flux0, col0, row0, sigma0

