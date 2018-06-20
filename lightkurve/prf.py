"""Provides callable models of the Kepler Pixel Response Function (PRF)."""
from __future__ import division, print_function

import math

from astropy.io import fits as pyfits
import numpy as np
import scipy
import scipy.interpolate

from .utils import channel_to_module_output, plot_image


__all__ = ['KeplerPRF', 'SimpleKeplerPRF']


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
        self.col_coord, self.row_coord, self.interpolate, self.supersampled_prf = self._prepare_prf()

    def __call__(self, flux, center_col, center_row, scale_col, scale_row,
                 rotation_angle):
        return self.evaluate(flux, center_col, center_row,
                             scale_col, scale_row, rotation_angle)

    def evaluate(self, flux, center_col, center_row, scale_col=1., scale_row=1.,
                 rotation_angle=0.):
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
                                                 rot_col.flatten() * scale_col,
                                                 grid=False).reshape(self.shape)
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
            prfn[i], crval1p[i], crval2p[i], cdelt1p[i], cdelt2p[i] = self._read_prf_calibration_file(
                prffile, i+1)

        prfn = np.array(prfn)
        PRFcol = np.arange(0.5, np.shape(prfn[0])[1] + 0.5)
        PRFrow = np.arange(0.5, np.shape(prfn[0])[0] + 0.5)
        PRFcol = (PRFcol - np.size(PRFcol) / 2) * cdelt1p[0]
        PRFrow = (PRFrow - np.size(PRFrow) / 2) * cdelt2p[0]

        # interpolate the calibrated PRF shape to the target position
        rowdim, coldim = self.shape[0], self.shape[1]
        prf = np.zeros(np.shape(prfn[0]), dtype='float32')
        ref_column = self.column + .5 * coldim
        ref_row = self.row + .5 * rowdim

        for i in range(n_hdu):
            prf_weight = math.sqrt((ref_column - crval1p[i]) ** 2
                                   + (ref_row - crval2p[i]) ** 2)
            if prf_weight < min_prf_weight:
                prf_weight = min_prf_weight
            prf += prfn[i] / prf_weight

        prf /= (np.nansum(prf) * cdelt1p[0] * cdelt2p[0])

        # location of the data image centered on the PRF image (in PRF pixel units)
        col_coord = np.arange(self.column + .5, self.column + coldim + .5)
        row_coord = np.arange(self.row + .5, self.row + rowdim + .5)
        # x-axis correspond to row-axis in scipy.RectBivariate
        # not to be confused with our convention, in which the
        # x-axis correspond to the column-axis
        interpolate = scipy.interpolate.RectBivariateSpline(PRFrow, PRFcol, prf)

        return col_coord, row_coord, interpolate, prf

    def plot(self, *params, **kwargs):
        pflux = self.evaluate(*params)
        plot_image(pflux, title='Kepler PRF Model, Channel: {}'.format(self.channel),
                   extent=(self.column, self.column + self.shape[1],
                           self.row, self.row + self.shape[0]), **kwargs)


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
