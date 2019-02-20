"""Provides callable models of the Kepler Pixel Response Function (PRF)."""
from __future__ import division, print_function

import math
from os.path import join, curdir
import urllib3
import certifi

from astropy.io import fits as pyfits
import numpy as np
import scipy.interpolate as scinterp
import scipy.io as scio

from ..utils import channel_to_module_output, plot_image
from ..search import default_download_dir

# Python2.7 doesnt define a FileNotFoundError
# so let's use the following hack:
# credit: https://stackoverflow.com/questions/21367320/searching-for-equivalent-of-filenotfounderror-in-python-2
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


__all__ = ['KeplerPRF', 'TessPRF', 'GaussianPRF']


class PRFModel:
    """A base class for a parametric model of a Pixel Response Function (PRF).

    Mathematically, the model is expressed as

    PRF(x, y) = flux * sig_x * sig_y * rot(phi(sig_x * (x - xc), sig_y * (y - yc)), alpha),
    where rot(., alpha) is the rotation operator, and alpha is angle of rotation.

    The parameters of the model are:
        - flux: the total flux of the source
        - sig_x, sig_y: PRF width factors in the x and y direction
        - xc, yc: coordinates of the center of the PRF
        - alpha: rotation angle

    The function phi represents an interpolation operation which is done on the basis
    of some calibrated/reference data.
    """
    def __call__(self, center_col, center_row, flux, scale_col, scale_row,
                 rotation_angle):
        return self.evaluate(center_col, center_row, flux,
                             scale_col, scale_row, rotation_angle)

    def evaluate(self, center_col, center_row, flux=1., scale_col=1., scale_row=1.,
                 rotation_angle=0.):
        """
        Interpolates the PRF model onto detector coordinates.

        Parameters
        ----------
        center_col, center_row : float
            Column and row coordinates of the center
        flux : float
            Total integrated flux of the PRF
        scale_col, scale_row : float
            Pixel scale stretch parameter, i.e. the numbers by which the PRF
            model needs to be multiplied in the column and row directions to
            account for focus changes
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

        delta_col = self.col_coord - (center_col + self.padding)
        delta_row = self.row_coord - (center_row + self.padding)
        delta_col, delta_row = np.meshgrid(delta_col, delta_row)

        rot_row = delta_row * cosa - delta_col * sina
        rot_col = delta_row * sina + delta_col * cosa

        modelshape = (self.shape[0] + self.padding*2, self.shape[1] + self.padding*2)
        prf_model = self.model(rot_row.flatten() * scale_row,
                               rot_col.flatten() * scale_col,
                               grid=False).reshape(modelshape)
        density = prf_model / np.nansum(prf_model)
        if self.padding == 0:  # TODO: May want to move this to constructor
            return flux * density
        return flux * density[self.padding:-self.padding, self.padding:-self.padding]

    def gradient(self, center_col, center_row, flux=1., scale_col=1., scale_row=1.,
                 rotation_angle=0.):
        """
        This function returns the gradient of the KeplerPRF model with
        respect to center_col, center_row, flux, scale_col, scale_row,
        and rotation_angle.

        Parameters
        ----------
        center_col, center_row : float
            Column and row coordinates of the center
        flux : float
            Total integrated flux of the PRF
        scale_col, scale_row : float
            Pixel scale stretch parameter, i.e. the numbers by which the PRF
            model needs to be multiplied in the column and row directions to
            account for focus changes
        rotation_angle : float
            Rotation angle in radians

        Returns
        -------
        grad_prf : list
            Returns a list of arrays where the elements are the partial derivatives
            of the KeplerPRF model with respect to center_col, center_row, flux, scale_col,
            scale_row, and rotation_angle, respectively.
        """
        raise NotImplementedError("Gradients are hard dude.")

    def plot(self, *params, title="PRF Model", **kwargs):
        pflux = self.evaluate(*params)
        plot_image(pflux, title=title,
                   extent=(self.column, self.column + self.shape[1],
                           self.row, self.row + self.shape[0]), **kwargs)


class KeplerPRF(PRFModel):
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

    def __init__(self, channel, shape, column, row, padding=10):
        self.channel = channel
        self.shape = shape
        self.column = column
        self.row = row
        self.padding = padding
        self.col_coord, self.row_coord, self.model, self.supersampled_prf = self._prepare_prf()
>>>>>>> 510604d7fcec77b24c7dc7b971a5b7917bceb178


class KeplerPRF(PRFModel):
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
        self.col_coord, self.row_coord, self.model, self.supersampled_prf = self._prepare_prf()

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
        rowdim, coldim = self.shape[0] + self.padding*2, self.shape[1] + self.padding*2
        prf = np.zeros(np.shape(prfn[0]), dtype='float32')
        ref_column = self.column + .5 * coldim + self.padding
        ref_row = self.row + .5 * rowdim + self.padding

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
        model = scinterp.RectBivariateSpline(PRFrow, PRFcol, prf)

        return col_coord, row_coord, model, prf

    def plot(self, *params, **kwargs):
        pflux = self.evaluate(*params)
        plot_image(pflux, title='Kepler PRF Model, Channel: {}'.format(self.channel),
                   extent=(self.column, self.column + self.shape[1],
                           self.row, self.row + self.shape[0]), **kwargs)


class TessPRF(PRFModel):
    """Builds a parametric PRF model on the basis of the calibrated PRF files
    made available by the TESS science team at https://archive.stsci.edu/missions/tess/models/.

    Parameters
    ----------
    camera : int
        The camera number.
    ccd : int
        The ccd number.
    shape : array-like of ints
        The size (number_of_rows, number_of_columns) of the tpf to be modelled.
    column, row: ints
        Column and row numbers of the bottom-left corner of the tpf.
    """
    def __init__(self, camera, ccd, shape, column, row, padding=10):
        self.camera = camera
        self.ccd = ccd
        self.shape = shape
        self.column = column
        self.row = row
        self.padding = padding
        self.col_coord, self.row_coord, self.model = self.build_model()

    def build_model(self):
        prf_struct = self.read_prf_file()
        prf_values, prf_row, prf_col = (prf_struct[0][0][0], prf_struct[0][0][5],
                                        prf_struct[0][0][6])
        prf_row = prf_row.flatten()
        prf_col = prf_col.flatten()
        row_diff = abs(prf_row[0] - prf_row[1])
        col_diff = abs(prf_col[0] - prf_col[1])
        prf_values /= (np.sum(prf_values) * row_diff * col_diff)
        rowdim, coldim = self.shape[0] + self.padding*2, self.shape[1] + self.padding*2
        col_coord = np.arange(self.column + .5, self.column + coldim + .5)
        row_coord = np.arange(self.row + .5, self.row + rowdim + .5)
        model = scinterp.RectBivariateSpline(prf_row, prf_col, prf_values)
        return col_coord, row_coord, model

    def read_prf_file(self):
        if (self.camera <= 2 and self.ccd <= 3):
            prefix = 'tess2018243163600-00072_035-'
        elif (self.camera == 1 and self.ccd == 4):
            prefix = 'tess2018243163600-00072_035-'
        else:
            prefix = 'tess2018243163601-00072_035-'
        filename = (prefix + str(self.camera) + '-' + str(self.ccd) +
                    '-characterized-prf.mat')
        url = 'https://archive.stsci.edu/missions/tess/models/' + filename
        # download .mat prf files and save them in the default
        # .lightkurve-cache directory
        tess_prf_dir = default_download_dir()
        file_path = join(tess_prf_dir, filename)
        try:
            prf_contents = scio.loadmat(file_path)
        except (FileNotFoundError, TypeError):
            http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED',
                                       ca_certs=certifi.where())
            response = http.request('GET', url)
            with open(file_path, 'wb') as f:
                f.write(response.data)
                prf_contents = scio.loadmat(file_path)
        return prf_contents['prfStruct']

    def plot(self, *params, **kwargs):
        pflux = self.evaluate(*params)
        plot_image(pflux, title='TESS PRF Model, Camera: {}, CCD: {}'.format(self.camera, self.ccd),
                   extent=(self.column, self.column + self.shape[1],
                           self.row, self.row + self.shape[0]), **kwargs)


class GaussianPRF(PRFModel):
    def __init__(self, shape, column, row, padding=5):
        self.shape = shape
        self.column = column
        self.row = row
        self.padding = padding
        rowdim, coldim = self.shape[0] + self.padding*2, self.shape[1] + self.padding*2
        self.x, self.y = np.meshgrid(np.arange(self.column + .5, self.column + coldim + .5),
                                     np.arange(self.row + .5, self.row + rowdim + .5))

    def evaluate(self, center_col, center_row, flux=1, scale_col=1, scale_row=1, rotation_angle=0):
        psi = rotation_angle
        a = .5 * ((scale_col * math.cos(psi)) ** 2 + (scale_row * math.sin(psi) ** 2))
        b = .25 * math.sin(2 * psi) * (scale_row ** 2 - scale_col ** 2)
        c = .5 * ((scale_col * math.sin(psi)) ** 2 + (scale_row * math.cos(psi) ** 2))
        xo, yo = center_col + self.padding, center_row + self.padding
        density = np.exp(-(a * (self.x - xo) ** 2 +
                           2 * b * (self.x - xo) * (self.y - yo) +
                           c * (self.y - yo) ** 2))
        result = flux * density / np.nansum(density)
        return result[self.padding:-self.padding, self.padding:-self.padding]

    def gradient(self):
        pass
