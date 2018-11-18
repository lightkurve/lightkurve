from __future__ import division
import datetime
import os
import warnings
import logging

from astropy.io import fits
from astropy.io.fits import Undefined
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from astropy.coordinates import SkyCoord
from astropy.stats.funcs import median_absolute_deviation as MAD

from . import PACKAGEDIR, MPLSTYLE
from .lightcurve import KeplerLightCurve, TessLightCurve
from .prf import KeplerPRF
from .utils import KeplerQualityFlags, TessQualityFlags, \
                   plot_image, bkjd_to_astropy_time, btjd_to_astropy_time, \
                   LightkurveWarning

__all__ = ['KeplerTargetPixelFile', 'TessTargetPixelFile']
log = logging.getLogger(__name__)


class TargetPixelFile(object):
    """
    Generic TargetPixelFile class for Kepler, K2, and TESS data.

    See `KeplerTargetPixelFile` and `TessTargetPixelFile` for constructor
    documentation.
    """
    def __init__(self, path, quality_bitmask='default', targetid=None, **kwargs):
        self.path = path
        if isinstance(path, fits.HDUList):
            self.hdu = path
        else:
            self.hdu = fits.open(self.path, **kwargs)
        self.quality_bitmask = quality_bitmask
        self.targetid = targetid

    def __getitem__(self, key):
        """Implements indexing and slicing.

        Note: the implementation below cannot be be simplified using
            `copy[1].data = copy[1].data[self.quality_mask][key]`
        due to the complicated behavior of AstroPy's `FITS_rec`.
        """
        # Step 1: determine the indexes of the data to return.
        # We start by determining the indexes of the good-quality cadences.
        quality_idx = np.where(self.quality_mask)[0]
        # Then we apply the index or slice to the good-quality indexes.
        if isinstance(key, int):
            # Ensure we always have a range; this is necessary to ensure
            # that we always ge a  `FITS_rec` instead of a `FITS_record` below.
            if key == -1:
                selected_idx = quality_idx[key:]
            else:
                selected_idx = quality_idx[key:key+1]
        else:
            selected_idx = quality_idx[key]

        # Step 2: use the indexes to create a new copy of the data.
        with warnings.catch_warnings():
            # Ignore warnings about empty fields
            warnings.simplefilter('ignore', UserWarning)
            # AstroPy added `HDUList.copy()` in v3.1, but we don't want to make
            # v3.1 a minimum requirement yet, so we copy in a funny way.
            copy = fits.HDUList([myhdu.copy() for myhdu in self.hdu])
            copy[1].data = copy[1].data[selected_idx]
        return self.__class__(copy, quality_bitmask=self.quality_bitmask, targetid=self.targetid)

    @property
    def hdu(self):
        return self._hdu

    @hdu.setter
    def hdu(self, value, keys=['FLUX', 'QUALITY']):
        '''Raises a ValueError exception if value does not appear to be a Target Pixel File.
        '''
        for key in keys:
            if ~(np.any([value[1].header[ttype] == key
                         for ttype in value[1].header['TTYPE*']])):
                raise ValueError("File {} does not have a {} column, "
                                 "is this a target pixel file?".format(self.path, key))
        else:
            self._hdu = value

    @property
    def header(self):
        """Returns the header of the primary extension."""
        return self.hdu[0].header

    @property
    def ra(self):
        """Right Ascension of target ('RA_OBJ' header keyword)."""
        try:
            return self.header['RA_OBJ']
        except KeyError:
            return None

    @property
    def dec(self):
        """Declination of target ('DEC_OBJ' header keyword)."""
        try:
            return self.header['DEC_OBJ']
        except KeyError:
            return None

    @property
    def column(self):
        try:
            out = self.hdu[1].header['1CRV5P']
        except KeyError:
            out = 0
        # ensure output has a value
        if isinstance(out, fits.card.Undefined):
            return 0
        else:
            return out

    @property
    def row(self):
        try:
            out = self.hdu[1].header['2CRV5P']
        except KeyError:
            out = 0
        # ensure output has a value
        if isinstance(out, fits.card.Undefined):
            return 0
        else:
            return out

    @property
    def pos_corr1(self):
        """Returns the column position correction."""
        return self.hdu[1].data['POS_CORR1'][self.quality_mask]

    @property
    def pos_corr2(self):
        """Returns the row position correction."""
        return self.hdu[1].data['POS_CORR2'][self.quality_mask]

    @property
    def pipeline_mask(self):
        """Returns the aperture mask used by the pipeline"""
        return self.hdu[2].data > 2

    @property
    def shape(self):
        """Return the cube dimension shape."""
        return self.flux.shape

    @property
    def time(self):
        """Returns the time for all good-quality cadences."""
        return self.hdu[1].data['TIME'][self.quality_mask]

    @property
    def cadenceno(self):
        """Return the cadence number for all good-quality cadences."""
        return self.hdu[1].data['CADENCENO'][self.quality_mask]

    @property
    def nan_time_mask(self):
        """Returns a boolean mask flagging cadences whose time is `nan`."""
        return ~np.isfinite(self.time)

    @property
    def flux(self):
        """Returns the flux for all good-quality cadences."""
        return self.hdu[1].data['FLUX'][self.quality_mask]

    @property
    def flux_err(self):
        """Returns the flux uncertainty for all good-quality cadences."""
        return self.hdu[1].data['FLUX_ERR'][self.quality_mask]

    @property
    def flux_bkg(self):
        """Returns the background flux for all good-quality cadences."""
        return self.hdu[1].data['FLUX_BKG'][self.quality_mask]

    @property
    def flux_bkg_err(self):
        return self.hdu[1].data['FLUX_BKG_ERR'][self.quality_mask]

    @property
    def quality(self):
        """Returns the quality flag integer of every good cadence."""
        return self.hdu[1].data['QUALITY'][self.quality_mask]

    @property
    def wcs(self):
        """Returns an astropy.wcs.WCS object with the World Coordinate System
        solution for the target pixel file.

        Returns
        -------
        w : astropy.wcs.WCS object
            WCS solution
        """
        # Use WCS keywords of the 5th column (FLUX)
        wcs_keywords = {'1CTYP5': 'CTYPE1',
                        '2CTYP5': 'CTYPE2',
                        '1CRPX5': 'CRPIX1',
                        '2CRPX5': 'CRPIX2',
                        '1CRVL5': 'CRVAL1',
                        '2CRVL5': 'CRVAL2',
                        '1CUNI5': 'CUNIT1',
                        '2CUNI5': 'CUNIT2',
                        '1CDLT5': 'CDELT1',
                        '2CDLT5': 'CDELT2',
                        '11PC5': 'PC1_1',
                        '12PC5': 'PC1_2',
                        '21PC5': 'PC2_1',
                        '22PC5': 'PC2_2',
                        'NAXIS1': 'NAXIS1',
                        'NAXIS2': 'NAXIS2'}
        mywcs = {}
        for oldkey, newkey in wcs_keywords.items():
            mywcs[newkey] = self.hdu[1].header[oldkey]
        return WCS(mywcs)

    @classmethod
    def from_fits(cls, path_or_url, **kwargs):
        """WARNING: THIS FUNCTION IS DEPRECATED AND WILL BE REMOVED VERY SOON.

        Please use `lightkurve.open()` instead.
        """
        warnings.warn('`TargetPixelFile.from_fits()` is deprecated and will be '
                      'removed soon, please use `lightkurve.open()` instead.',
                      LightkurveWarning)
        return cls(path_or_url, **kwargs)

    def get_coordinates(self, cadence='all'):
        """Returns two 3D arrays of RA and Dec values in decimal degrees.

        If cadence number is given, returns 2D arrays for that cadence. If
        cadence is 'all' returns one RA, Dec value for each pixel in every cadence.
        Uses the WCS solution and the POS_CORR data from TPF header.

        Parameters
        ----------
        cadence : 'all' or int
            Which cadences to return the RA Dec coordinates for.

        Returns
        -------
        ra : numpy array, same shape as tpf.flux[cadence]
            Array containing RA values for every pixel, for every cadence.
        dec : numpy array, same shape as tpf.flux[cadence]
            Array containing Dec values for every pixel, for every cadence.
        """
        w = self.wcs
        X, Y = np.meshgrid(np.arange(self.shape[2]), np.arange(self.shape[1]))
        pos_corr1_pix = np.copy(self.hdu[1].data['POS_CORR1'])
        pos_corr2_pix = np.copy(self.hdu[1].data['POS_CORR2'])

        # We zero POS_CORR* when the values are NaN or make no sense (>50px)
        with warnings.catch_warnings():  # Comparing NaNs to numbers is OK here
            warnings.simplefilter("ignore", RuntimeWarning)
            bad = np.any([~np.isfinite(pos_corr1_pix),
                          ~np.isfinite(pos_corr2_pix),
                          np.abs(pos_corr1_pix - np.nanmedian(pos_corr1_pix)) > 50,
                          np.abs(pos_corr2_pix - np.nanmedian(pos_corr2_pix)) > 50], axis=0)
        pos_corr1_pix[bad], pos_corr2_pix[bad] = 0, 0

        # Add in POSCORRs
        X = (np.atleast_3d(X).transpose([2, 0, 1]) +
             np.atleast_3d(pos_corr1_pix).transpose([1, 2, 0]))
        Y = (np.atleast_3d(Y).transpose([2, 0, 1]) +
             np.atleast_3d(pos_corr2_pix).transpose([1, 2, 0]))

        # Pass through WCS
        ra, dec = w.wcs_pix2world(X.ravel(), Y.ravel(), 1)
        ra = ra.reshape((pos_corr1_pix.shape[0], self.shape[1], self.shape[2]))
        dec = dec.reshape((pos_corr2_pix.shape[0], self.shape[1], self.shape[2]))
        ra, dec = ra[self.quality_mask], dec[self.quality_mask]
        if cadence is not 'all':
            return ra[cadence], dec[cadence]
        return ra, dec

    def show_properties(self):
        '''Print out a description of each of the non-callable attributes of a
        TargetPixelFile object.

        Prints in order of type (ints, strings, lists, arrays and others)
        Prints in alphabetical order.'''
        attrs = {}
        for attr in dir(self):
            if not attr.startswith('_'):
                res = getattr(self, attr)
                if callable(res):
                    continue
                if attr == 'hdu':
                    attrs[attr] = {'res': res, 'type': 'list'}
                    for idx, r in enumerate(res):
                        if idx == 0:
                            attrs[attr]['print'] = '{}'.format(r.header['EXTNAME'])
                        else:
                            attrs[attr]['print'] = '{}, {}'.format(attrs[attr]['print'],
                                                                   '{}'.format(r.header['EXTNAME']))
                    continue
                else:
                    attrs[attr] = {'res': res}
                if isinstance(res, int):
                    attrs[attr]['print'] = '{}'.format(res)
                    attrs[attr]['type'] = 'int'
                elif isinstance(res, np.ndarray):
                    attrs[attr]['print'] = 'array {}'.format(res.shape)
                    attrs[attr]['type'] = 'array'
                elif isinstance(res, list):
                    attrs[attr]['print'] = 'list length {}'.format(len(res))
                    attrs[attr]['type'] = 'list'
                elif isinstance(res, str):
                    if res == '':
                        attrs[attr]['print'] = '{}'.format('None')
                    else:
                        attrs[attr]['print'] = '{}'.format(res)
                    attrs[attr]['type'] = 'str'
                elif attr == 'wcs':
                    attrs[attr]['print'] = 'astropy.wcs.wcs.WCS'.format(attr)
                    attrs[attr]['type'] = 'other'
                else:
                    attrs[attr]['print'] = '{}'.format(type(res))
                    attrs[attr]['type'] = 'other'
        output = Table(names=['Attribute', 'Description'], dtype=[object, object])
        idx = 0
        types = ['int', 'str', 'list', 'array', 'other']
        for typ in types:
            for attr, dic in attrs.items():
                if dic['type'] == typ:
                    output.add_row([attr, dic['print']])
                    idx += 1
        output.pprint(max_lines=-1, max_width=-1)

    def to_lightcurve(self, method='aperture', **kwargs):
        """Performs photometry.

        See the docstring of `aperture_photometry()` for valid
        arguments if the method is 'aperture'.  Otherwise, see the docstring
        of `prf_photometry()` for valid arguments if the method is 'prf'.

        Parameters
        ----------
        method : 'aperture' or 'prf'.
            Photometry method to use.
        **kwargs : dict
            Extra arguments to be passed to the `aperture_photometry` or the
            `prf_photometry` method of this class.

        Returns
        -------
        lc : LightCurve object
            Object containing the resulting lightcurve.
        """
        if method == 'aperture':
            return self.extract_aperture_photometry(**kwargs)
        elif method == 'prf':
            return self.prf_lightcurve(**kwargs)
        else:
            raise ValueError("Photometry method must be 'aperture' or 'prf'.")

    def _parse_aperture_mask(self, aperture_mask):
        """Parse the `aperture_mask` parameter as given by a user.

        The `aperture_mask` parameter is accepted by a number of methods.
        This method ensures that the parameter is always parsed in the same way.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', 'all', 'threshold', or None
            A boolean array describing the aperture such that `True` means
            that the pixel will be used.
            If None or 'all' are passed, all pixels will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.

        Returns
        -------
        aperture_mask : ndarray
            2D boolean numpy array containing `True` for selected pixels.
        """
        with warnings.catch_warnings():
            # `aperture_mask` supports both arrays and string values; these yield
            # uninteresting FutureWarnings when compared, so let's ignore that.
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if aperture_mask is None or aperture_mask == 'all':
                aperture_mask = np.ones((self.shape[1], self.shape[2]), dtype=bool)
            elif aperture_mask == 'pipeline':
                aperture_mask = self.pipeline_mask
            elif aperture_mask == 'threshold':
                aperture_mask = self.create_threshold_mask()
        self._last_aperture_mask = aperture_mask
        return aperture_mask

    def create_threshold_mask(self, threshold=3):
        """Returns an aperture mask creating using the thresholding method.

        This method will identify the pixels in the TargetPixelFile which show
        a median flux that is brighter than `threshold` times the standard
        deviation above the overall median. The standard deviation is estimated
        in a robust way by multiplying the Median Absolute Deviation (MAD)
        with 1.4826.

        Parameters
        ----------
        threshold : float
            A value for the number of sigma by which a pixel needs to be
            brighter than the median flux to be included in the aperture mask.

        Returns
        -------
        aperture_mask : ndarray
            2D boolean numpy array containing `True` for pixels above the
            threshold.
        """
        # Calculate the median image
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            median_image = np.nanmedian(self.flux, axis=0)
        vals = median_image[np.isfinite(median_image)].flatten()
        # Calculate the theshold value in flux units
        mad_cut = (1.4826 * MAD(vals) * threshold) + np.nanmedian(median_image)
        # Create a mask containing the pixels above the threshold flux
        return np.nan_to_num(median_image) > mad_cut

    def centroids(self, **kwargs):
        """DEPRECATED: use `estimate_cdpp()` instead."""
        warnings.warn('`TargetPixelFile.centroids()` is deprecated and will be '
                      'removed in Lightkurve v1.0.0, '
                      'please use `TargetPixelFile.estimate_centroids()` instead.',
                      LightkurveWarning)
        return self.estimate_centroids(**kwargs)

    def estimate_centroids(self, aperture_mask='pipeline'):
        """Returns centroid positions estimated using sample moments.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', or 'all'
            A boolean array describing the aperture such that `True` means
            that the pixel will be used.
            If None or 'all' are passed, all pixels will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.

        Returns
        -------
        col_centr, row_centr : tuple
            Arrays containing centroids for column and row at each cadence
        """
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        yy, xx = np.indices(self.shape[1:]) + 0.5
        yy = self.row + yy
        xx = self.column + xx
        total_flux = np.nansum(self.flux[:, aperture_mask], axis=1)
        with warnings.catch_warnings():
            # RuntimeWarnings may occur below if total_flux contains zeros
            warnings.simplefilter("ignore", RuntimeWarning)
            col_centr = np.nansum(xx * aperture_mask * self.flux, axis=(1, 2)) / total_flux
            row_centr = np.nansum(yy * aperture_mask * self.flux, axis=(1, 2)) / total_flux
        return col_centr, row_centr

    def plot(self, ax=None, frame=0, cadenceno=None, bkg=False, aperture_mask=None,
             show_colorbar=True, mask_color='pink', style='lightkurve', **kwargs):
        """
        Plot a target pixel file at a given frame (index) or cadence number.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        frame : int
            Frame number. The default is 0, i.e. the first frame.
        cadenceno : int, optional
            Alternatively, a cadence number can be provided.
            This argument has priority over frame number.
        bkg : bool
            If True, background will be added to the pixel values.
        aperture_mask : ndarray or str
            Highlight pixels selected by aperture_mask.
        show_colorbar : bool
            Whether or not to show the colorbar
        mask_color : str
            Color to show the aperture mask
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Keywords arguments passed to `lightkurve.utils.plot_image`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if style == 'lightkurve' or style is None:
            style = MPLSTYLE
        if cadenceno is not None:
            try:
                frame = np.argwhere(cadenceno == self.cadenceno)[0][0]
            except IndexError:
                raise ValueError("cadenceno {} is out of bounds, "
                                 "must be in the range {}-{}.".format(
                                     cadenceno, self.cadenceno[0], self.cadenceno[-1]))
        try:
            if bkg and np.any(np.isfinite(self.flux_bkg[frame])):
                pflux = self.flux[frame] + self.flux_bkg[frame]
            else:
                pflux = self.flux[frame]
        except IndexError:
            raise ValueError("frame {} is out of bounds, must be in the range "
                             "0-{}.".format(frame, self.shape[0]))
        with plt.style.context(style):
            img_title = 'Target ID: {}'.format(self.targetid)
            img_extent = (self.column, self.column + self.shape[2],
                          self.row, self.row + self.shape[1])
            ax = plot_image(pflux, ax=ax, title=img_title, extent=img_extent,
                            show_colorbar=show_colorbar, **kwargs)
            ax.grid(False)
        if aperture_mask is not None:
            aperture_mask = self._parse_aperture_mask(aperture_mask)
            for i in range(self.shape[1]):
                for j in range(self.shape[2]):
                    if aperture_mask[i, j]:
                        ax.add_patch(patches.Rectangle((j+self.column, i+self.row),
                                                       1, 1, color=mask_color, fill=True,
                                                       alpha=.6))
        return ax

    def to_fits(self, output_fn=None, overwrite=False):
        """Writes the TPF to a FITS file on disk."""
        if output_fn is None:
            output_fn = "{}-targ.fits".format(self.targetid)
        self.hdu.writeto(output_fn, overwrite=overwrite, checksum=True)

    def interact(self, lc=None, notebook_url='localhost:8888', max_cadences=30000):
        """Display an interactive Jupyter Notebook widget to inspect the pixel data.

        The widget will show both the lightcurve and pixel data.  By default,
        the lightcurve shown is obtained by calling the `to_lightcurve()` method,
        unless the user supplies a custom `LightCurve` object.
        This feature requires an optional dependency, bokeh (v0.12.15 or later).
        This dependency can be installed using e.g. `conda install bokeh`.

        At this time, this feature only works inside an active Jupyter
        Notebook, and tends to be too slow when more than ~30,000 cadences
        are contained in the TPF (e.g. short cadence data).

        Parameters
        ----------
        lc : LightCurve object
            An optional pre-processed lightcurve object to show.
        notebook_url: str
            Location of the Jupyter notebook page (default: "localhost:8888")
            When showing Bokeh applications, the Bokeh server must be
            explicitly configured to allow connections originating from
            different URLs. This parameter defaults to the standard notebook
            host and port. If you are running on a different location, you
            will need to supply this value for the application to display
            properly. If no protocol is supplied in the URL, e.g. if it is
            of the form "localhost:8888", then "http" will be used.
        max_cadences : int
            Raise a RuntimeError if the number of cadences shown is larger than
            this value. This limit helps keep browsers from becoming unresponsive.
        """
        from .interact import show_interact_widget
        return show_interact_widget(self, lc=lc, notebook_url=notebook_url,
                                    max_cadences=max_cadences)


class KeplerTargetPixelFile(TargetPixelFile):
    """
    Defines a TargetPixelFile class for the Kepler/K2 Mission.
    Enables extraction of raw lightcurves and centroid positions.

    Parameters
    ----------
    path : str or `astropy.io.fits.HDUList`
        Path to a Kepler Target Pixel (FITS) File or a `HDUList` object.
    quality_bitmask : str or int
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. If a string is passed, it has the
        following meaning:

            * "none": no cadences will be ignored (`quality_bitmask=0`).
            * "default": cadences with severe quality issues will be ignored
              (`quality_bitmask=1130799`).
            * "hard": more conservative choice of flags to ignore
              (`quality_bitmask=1664431`). This is known to remove good data.
            * "hardest": removes all data that has been flagged
              (`quality_bitmask=2096639`). This mask is not recommended.

        See the :class:`KeplerQualityFlags` class for details on the bitmasks.
    kwargs : dict
        Keyword arguments passed to `astropy.io.fits.open()`.

    References
    ----------
    .. [1] Kepler: A Search for Terrestrial Planets. Kepler Archive Manual.
        http://archive.stsci.edu/kepler/manuals/archive_manual.pdf
    """
    def __init__(self, path, quality_bitmask='default', **kwargs):
        super(KeplerTargetPixelFile, self).__init__(path,
                                                    quality_bitmask=quality_bitmask,
                                                    **kwargs)
        self.quality_mask = KeplerQualityFlags.create_quality_mask(
                                quality_array=self.hdu[1].data['QUALITY'],
                                bitmask=quality_bitmask)

        # Check the TELESCOP keyword and warn the user if it's not 'Kepler'
        try:
            telescop = self.header['telescop']
            if telescop == 'TESS':
                warnings.warn("A TESS data product is being opened using the "
                              "`KeplerTargetPixelFile` class. "
                              "Please use `TessTargetPixelFile` instead.",
                              LightkurveWarning)
            elif telescop != 'Kepler':
                warnings.warn("KeplerTargetPixelFile encountered 'TELESCOP' "
                              "keyword '{}' instead of 'Kepler'".format(telescop),
                              LightkurveWarning)
        except KeyError:
            log.debug("KeplerTargetPixelFile encountered a file without 'TELESCOP' keyword.")

        # Use the KEPLERID keyword as the default targetid
        if self.targetid is None:
            try:
                self.targetid = self.header['KEPLERID']
            except KeyError:
                pass

    @staticmethod
    def from_archive(target, cadence='long', quarter=None, month=None,
                     campaign=None, quality_bitmask='default', **kwargs):
        """WARNING: THIS FUNCTION IS DEPRECATED AND WILL BE REMOVED VERY SOON.
        Use `lightkurve.search_targetpixelfile()` instead.

        Parameters
        ----------
        target : str or int
            KIC/EPIC ID or object name.
        cadence : str
            'long' or 'short'.
        quarter, campaign : int, list of ints, or 'all'
            Kepler Quarter or K2 Campaign number.
        month : 1, 2, 3, list or 'all'
            For Kepler's prime mission, there are three short-cadence
            Target Pixel Files for each quarter, each covering one month.
            Hence, if cadence='short' you need to specify month=1, 2, or 3.
        quality_bitmask : str or int
            Bitmask (integer) which identifies the quality flag bitmask that should
            be used to mask out bad cadences. If a string is passed, it has the
            following meaning:

                * "none": no cadences will be ignored (`quality_bitmask=0`).
                * "default": cadences with severe quality issues will be ignored
                  (`quality_bitmask=1130799`).
                * "hard": more conservative choice of flags to ignore
                  (`quality_bitmask=1664431`). This is known to remove good data.
                * "hardest": removes all data that has been flagged
                  (`quality_bitmask=2096639`). This mask is not recommended.

            See the :class:`KeplerQualityFlags` class for details on the bitmasks.
        kwargs : dict
            Keywords arguments passed to the constructor of
            :class:`KeplerTargetPixelFile`.

        Returns
        -------
        tpf : :class:`KeplerTargetPixelFile` or :class:`TargetPixelFileCollection` object.
        """
        warnings.warn('`TargetPixelFile.from_archive` is deprecated and will be removed soon, '
                      'please use `lightkurve.search_targetpixelfile()` instead.',
                      LightkurveWarning)
        if os.path.exists(str(target)) or str(target).startswith('http'):
            log.warning('Warning: from_archive() is not intended to accept a '
                        'direct path, use KeplerTargetPixelFile(path) instead.')
            return KeplerTargetPixelFile(target)
        else:
            from .search import search_targetpixelfile
            sr = search_targetpixelfile(target, cadence=cadence,
                                        quarter=quarter, month=month,
                                        campaign=campaign)
            if len(sr) == 1:
                return sr.download(quality_bitmask=quality_bitmask, **kwargs)
            elif len(sr) > 1:
                return sr.download_all(quality_bitmask=quality_bitmask, **kwargs)
            else:
                raise ValueError("No target pixel files found that match the search criteria.")

    def __repr__(self):
        return('KeplerTargetPixelFile Object (ID: {})'.format(self.targetid))

    def get_prf_model(self):
        """Returns an object of KeplerPRF initialized using the
        necessary metadata in the tpf object.

        Returns
        -------
        prf : instance of SimpleKeplerPRF
        """

        return KeplerPRF(channel=self.channel, shape=self.shape[1:],
                         column=self.column, row=self.row)

    @property
    def obsmode(self):
        """'short cadence' or 'long cadence'. ('OBSMODE' header keyword)"""
        return self.header['OBSMODE']

    @property
    def module(self):
        """Kepler CCD module number. ('MODULE' header keyword)"""
        return self.header['MODULE']

    @property
    def output(self):
        """Kepler CCD module output number. ('OUTPUT' header keyword)"""
        return self.header['OUTPUT']

    @property
    def channel(self):
        """Kepler CCD channel number. ('CHANNEL' header keyword)"""
        return self.header['CHANNEL']

    @property
    def astropy_time(self):
        """Returns an AstroPy Time object for all good-quality cadences."""
        return bkjd_to_astropy_time(bkjd=self.time)

    @property
    def quarter(self):
        """Kepler quarter number. ('QUARTER' header keyword)"""
        try:
            return self.header['QUARTER']
        except KeyError:
            return None

    @property
    def campaign(self):
        """K2 Campaign number. ('CAMPAIGN' header keyword)"""
        try:
            return self.header['CAMPAIGN']
        except KeyError:
            return None

    @property
    def mission(self):
        """'Kepler' or 'K2'. ('MISSION' header keyword)"""
        try:
            return self.header['MISSION']
        except KeyError:
            return None

    def extract_aperture_photometry(self, aperture_mask='pipeline'):
        """Returns a LightCurve obtained using aperture photometry.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', 'threshold' or 'all'
            A boolean array describing the aperture such that `True` means
            that the pixel will be used.
            If None or 'all' are passed, all pixels will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.

        Returns
        -------
        lc : KeplerLightCurve object
            Array containing the summed flux within the aperture for each
            cadence.
        """
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        if aperture_mask.sum() == 0:
            log.warning('Warning: aperture mask contains zero pixels.')
        centroid_col, centroid_row = self.estimate_centroids(aperture_mask)
        # Ignore warnings related to zero or negative errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            flux_err = np.nansum(self.flux_err[:, aperture_mask]**2, axis=1)**0.5

        keys = {'centroid_col': centroid_col,
                'centroid_row': centroid_row,
                'quality': self.quality,
                'channel': self.channel,
                'campaign': self.campaign,
                'quarter': self.quarter,
                'mission': self.mission,
                'cadenceno': self.cadenceno,
                'ra': self.ra,
                'dec': self.dec,
                'label': self.header['OBJECT'],
                'targetid': self.targetid}
        return KeplerLightCurve(time=self.time,
                                time_format='bkjd',
                                time_scale='tdb',
                                flux=np.nansum(self.flux[:, aperture_mask], axis=1),
                                flux_err=flux_err,
                                **keys)

    def get_bkg_lightcurve(self, aperture_mask=None):
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        # Ignore warnings related to zero or negative errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            flux_bkg_err = np.nansum(self.flux_bkg_err[:, aperture_mask]**2, axis=1)**0.5
        keys = {'quality': self.quality,
                'channel': self.channel,
                'campaign': self.campaign,
                'quarter': self.quarter,
                'mission': self.mission,
                'cadenceno': self.cadenceno,
                'ra': self.ra,
                'dec': self.dec,
                'label': self.header['OBJECT'],
                'targetid': self.targetid}
        return KeplerLightCurve(time=self.time,
                                time_format='bkjd',
                                time_scale='tdb',
                                flux=np.nansum(self.flux_bkg[:, aperture_mask], axis=1),
                                flux_err=flux_bkg_err,
                                **keys)

    def get_model(self, star_priors=None, **kwargs):
        """Returns a default `TPFModel` object for PRF fitting.

        The default model only includes one star and only allows its flux
        and position to change.  A different set of stars can be added using
        the `star_priors` parameter.

        Parameters
        ----------
        **kwargs : dict
            Arguments to be passed to the `TPFModel` constructor, e.g.
            `star_priors`.

        Returns
        -------
        model : TPFModel object
            Model with appropriate defaults for this Target Pixel File.
        """
        from .prf import TPFModel, StarPrior, BackgroundPrior
        from .prf import UniformPrior, GaussianPrior
        # Set up the model
        if 'star_priors' not in kwargs:
            centr_col, centr_row = self.estimate_centroids()
            star_priors = [StarPrior(col=GaussianPrior(mean=np.nanmedian(centr_col),
                                                       var=np.nanstd(centr_col)**2),
                                     row=GaussianPrior(mean=np.nanmedian(centr_row),
                                                       var=np.nanstd(centr_row)**2),
                                     flux=UniformPrior(lb=0.5*np.nanmax(self.flux[0]),
                                                       ub=2*np.nansum(self.flux[0]) + 1e-10),
                                     targetid=self.targetid)]
            kwargs['star_priors'] = star_priors
        if 'prfmodel' not in kwargs:
            kwargs['prfmodel'] = self.get_prf_model()
        if 'background_prior' not in kwargs:
            if np.all(np.isnan(self.flux_bkg)):  # If TargetPixelFile has no background flux data
                # Use the median of the lower half of flux as an estimate for flux_bkg
                clipped_flux = np.ma.masked_where(self.flux > np.percentile(self.flux, 50),
                                                  self.flux)
                flux_prior = GaussianPrior(mean=np.ma.median(clipped_flux),
                                           var=np.ma.std(clipped_flux)**2)
            else:
                flux_prior = GaussianPrior(mean=np.nanmedian(self.flux_bkg),
                                           var=np.nanstd(self.flux_bkg)**2)
            kwargs['background_prior'] = BackgroundPrior(flux=flux_prior)
        return TPFModel(**kwargs)

    def extract_prf_photometry(self, cadences=None, parallel=True, **kwargs):
        """Returns the results of PRF photometry applied to the pixel file.

        Parameters
        ----------
        cadences : list of int
            Cadences to fit.  If `None` (default) then all cadences will be fit.
        parallel : bool
            If `True`, fitting cadences will be distributed across multiple
            cores using Python's `multiprocessing` module.
        **kwargs : dict
            Keywords to be passed to `tpf.get_model()` to create the
            `TPFModel` object that will be fit.

        Returns
        -------
        results : PRFPhotometry object
            Object that provides access to PRF-fitting photometry results and
            various diagnostics.
        """
        from .prf import PRFPhotometry
        log.warning('Warning: PRF-fitting photometry is experimental '
                    'in this version of lightkurve.')
        prfphot = PRFPhotometry(model=self.get_model(**kwargs))
        prfphot.run(self.flux + self.flux_bkg, cadences=cadences, parallel=parallel,
                    pos_corr1=self.pos_corr1, pos_corr2=self.pos_corr2)
        return prfphot

    def prf_lightcurve(self, **kwargs):
        lc = self.extract_prf_photometry(**kwargs).lightcurves[0]
        keys = {'quality': self.quality,
                'channel': self.channel,
                'campaign': self.campaign,
                'quarter': self.quarter,
                'mission': self.mission,
                'cadenceno': self.cadenceno,
                'ra': self.ra,
                'dec': self.dec,
                'targetid': self.targetid}
        return KeplerLightCurve(time=self.time,
                                flux=lc.flux,
                                time_format='bkjd',
                                time_scale='tdb',
                                **keys)

    @staticmethod
    def from_fits_images(images, position, size=(11, 11), extension=1,
                         target_id="unnamed-target", hdu0_keywords={}, **kwargs):
        """Creates a new Target Pixel File from a set of images.

        This method is intended to make it easy to cut out targets from
        Kepler/K2 "superstamp" regions or TESS FFI images.

        Parameters
        ----------
        images : list of str, or list of fits.ImageHDU objects
            Sorted list of FITS filename paths or ImageHDU objects to get
            the data from.
        position : astropy.SkyCoord
            Position around which to cut out pixels.
        size : (int, int)
            Dimensions (cols, rows) to cut out around `position`.
        extension : int or str
            If `images` is a list of filenames, provide the extension number
            or name to use. Default: 0.
        target_id : int or str
            Unique identifier of the target to be recorded in the TPF.
        hdu0_keywords : dict
            Additional keywords to add to the first header file.
        **kwargs : dict
            Extra arguments to be passed to the `KeplerTargetPixelFile` constructor.

        Returns
        -------
        tpf : KeplerTargetPixelFile
            A new Target Pixel File assembled from the images.
        """
        if len(images) == 0:
            raise ValueError('One or more images must be passed.')
        if not isinstance(position, SkyCoord):
            raise ValueError('Position must be an astropy.coordinates.SkyCoord.')

        basic_keywords = ['MISSION', 'TELESCOP', 'INSTRUME', 'QUARTER',
                          'CAMPAIGN', 'CHANNEL', 'MODULE', 'OUTPUT']
        carry_keywords = {}

        # Define a helper function to accept images in a flexible way
        def _open_image(img, extension):
            if isinstance(img, fits.ImageHDU):
                hdu = img
            elif isinstance(img, fits.HDUList):
                hdu = img[extension]
            else:
                hdu = fits.open(img)[extension]
            return hdu

        # Set the default extension if unspecified
        if extension is None:
            extension = 0
            if isinstance(images[0], str) and images[0].endswith("ffic.fits"):
                extension = 1  # TESS FFIs have the image data in extension #1

        # If no position is given, ensure the cut-out size matches the image size
        if position is None:
            size = _open_image(images[0], extension).data.shape

        # Find middle image to use as a WCS reference
        try:
            mid_hdu = _open_image(images[int(len(images) / 2) - 1], extension)
            wcs_ref = WCS(mid_hdu)
            column, row = wcs_ref.wcs_world2pix(
                            np.asarray([[position.ra.deg], [position.dec.deg]]).T,
                            0)[0]
            column, row = int(column), int(row)
        except Exception as e:
            raise e
            raise FactoryError("Images must have a valid WCS astrometric solution.")
            return None

        # Create a factory and set default keyword values based on the middle image
        factory = KeplerTargetPixelFileFactory(n_cadences=len(images),
                                               n_rows=size[0],
                                               n_cols=size[1],
                                               target_id=target_id)

        # Get some basic keywords
        for kw in basic_keywords:
            if kw in mid_hdu.header:
                if not isinstance(mid_hdu.header[kw], Undefined):
                    carry_keywords[kw] = mid_hdu.header[kw]
        if ('MISSION' not in carry_keywords) and ('TELESCOP' in carry_keywords):
            carry_keywords['MISSION'] = carry_keywords['TELESCOP']

        allkeys = hdu0_keywords.copy()
        allkeys.update(carry_keywords)

        ext_info = {'1CRV5P': column, '2CRV5P': row}

        for idx, img in tqdm(enumerate(images), total=len(images)):
            hdu = _open_image(img, extension)

            if idx == 0:  # Get default keyword values from the first image
                factory.keywords = hdu.header
            if position is None:
                cutout = hdu
            else:
                cutout = Cutout2D(hdu.data, position, wcs=WCS(hdu.header),
                                  size=size, mode='partial')
            factory.add_cadence(frameno=idx, flux=cutout.data, header=hdu.header)
        return factory.get_tpf(hdu0_keywords=allkeys, ext_info=ext_info, **kwargs)


class FactoryError(Exception):
    """Raised if there is a problem creating a TPF."""
    pass


class KeplerTargetPixelFileFactory(object):
    """Class to create a KeplerTargetPixelFile."""

    def __init__(self, n_cadences, n_rows, n_cols, target_id="unnamed-target",
                 keywords={}):
        self.n_cadences = n_cadences
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.target_id = target_id
        self.keywords = keywords

        # Initialize the 3D data structures
        self.raw_cnts = np.empty((n_cadences, n_rows, n_cols), dtype='int')
        self.flux = np.empty((n_cadences, n_rows, n_cols), dtype='float32')
        self.flux_err = np.empty((n_cadences, n_rows, n_cols), dtype='float32')
        self.flux_bkg = np.empty((n_cadences, n_rows, n_cols), dtype='float32')
        self.flux_bkg_err = np.empty((n_cadences, n_rows, n_cols), dtype='float32')
        self.cosmic_rays = np.empty((n_cadences, n_rows, n_cols), dtype='float32')

        # Set 3D data defaults
        self.raw_cnts[:, :, :] = -1
        self.flux[:, :, :] = np.nan
        self.flux_err[:, :, :] = np.nan
        self.flux_bkg[:, :, :] = np.nan
        self.flux_bkg_err[:, :, :] = np.nan
        self.cosmic_rays[:, :, :] = np.nan

        # Initialize the 1D data structures
        self.mjd = np.zeros(n_cadences, dtype='float64')
        self.time = np.zeros(n_cadences, dtype='float64')
        self.timecorr = np.zeros(n_cadences, dtype='float32')
        self.cadenceno = np.zeros(n_cadences, dtype='int')
        self.quality = np.zeros(n_cadences, dtype='int')
        self.pos_corr1 = np.zeros(n_cadences, dtype='float32')
        self.pos_corr2 = np.zeros(n_cadences, dtype='float32')

    def add_cadence(self, frameno, wcs=None, raw_cnts=None, flux=None, flux_err=None,
                    flux_bkg=None, flux_bkg_err=None, cosmic_rays=None,
                    header={}):
        """Populate the data for a single cadence."""
        if frameno >= self.n_cadences:
            raise FactoryError('Can not add cadence {}, n_cadences set to {}'.format(frameno, self.n_cadences))

        # 2D-data
        for col in ['raw_cnts', 'flux', 'flux_err', 'flux_bkg',
                    'flux_bkg_err', 'cosmic_rays']:
            if locals()[col] is not None:
                if locals()[col].shape != (self.n_rows, self.n_cols):
                    raise FactoryError('Can not add cadence with a different shape ({} x {})'.format(self.n_rows, self.n_cols))

                vars(self)[col][frameno] = locals()[col]

        # 1D-data
        if 'TSTART' in header and 'TSTOP' in header:
            self.time[frameno] = (header['TSTART'] + header['TSTOP']) / 2.
        if 'TIMECORR' in header:
            self.timecorr[frameno] = header['TIMECORR']
        if 'CADENCEN' in header:
            self.cadenceno[frameno] = header['CADENCEN']
        if 'QUALITY' in header:
            self.quality[frameno] = header['QUALITY']
        if 'POS_CORR1' in header:
            self.pos_corr1[frameno] = header['POS_CORR1']
        if 'POS_CORR2' in header:
            self.pos_corr2[frameno] = header['POS_CORR2']
        if wcs is None:
            self.pos_corr1[frameno], self.pos_corr2[frameno] = None, None

    def _check_data(self):
        ''' Check the data before writing to a TPF for any obvious errors
        '''
        if len(self.time) != len(np.unique(self.time)):
            warnings.warn('The factory-created TPF contains cadences with '
                          'identical TIME values.', LightkurveWarning)
        if ~np.all(self.time == np.sort(self.time)):
            warnings.warn('Cadences in the factory-created TPF do not appear '
                          'to be sorted in chronological order.', LightkurveWarning)
        if np.nansum(self.flux) == 0:
            warnings.warn('The factory-created TPF does not appear to contain '
                          'non-zero flux values.', LightkurveWarning)

    def get_tpf(self, hdu0_keywords={}, ext_info={}, **kwargs):
        """Returns a KeplerTargetPixelFile object."""
        self._check_data()
        return KeplerTargetPixelFile(self._hdulist(hdu0_keywords=hdu0_keywords,
                                                   ext_info=ext_info),
                                     **kwargs)

    def _hdulist(self, hdu0_keywords={}, ext_info={}):
        """Returns an astropy.io.fits.HDUList object."""
        return fits.HDUList([self._make_primary_hdu(hdu0_keywords=hdu0_keywords),
                             self._make_target_extension(ext_info=ext_info),
                             self._make_aperture_extension()])

    def _header_template(self, extension):
        """Returns a template `fits.Header` object for a given extension."""
        template_fn = os.path.join(PACKAGEDIR, "data",
                                   "tpf-ext{}-header.txt".format(extension))
        return fits.Header.fromtextfile(template_fn)

    def _make_primary_hdu(self, hdu0_keywords={}):
        """Returns the primary extension (#0)."""
        hdu = fits.PrimaryHDU()
        # Copy the default keywords from a template file from the MAST archive
        tmpl = self._header_template(0)
        for kw in tmpl:
            hdu.header[kw] = (tmpl[kw], tmpl.comments[kw])
        # Override the defaults where necessary
        hdu.header['ORIGIN'] = "Unofficial data product"
        hdu.header['DATE'] = datetime.datetime.now().strftime("%Y-%m-%d")
        hdu.header['CREATOR'] = "lightkurve"
        hdu.header['OBJECT'] = self.target_id
        hdu.header['KEPLERID'] = self.target_id
        # Empty a bunch of keywords rather than having incorrect info
        for kw in ["PROCVER", "FILEVER", "CHANNEL", "MODULE", "OUTPUT",
                   "TIMVERSN", "CAMPAIGN", "DATA_REL", "TTABLEID",
                   "RA_OBJ", "DEC_OBJ"]:
            hdu.header[kw] = ""

        # Some keywords just shouldn't be passed to the new header.
        bad_keys = ['ORIGIN', 'DATE', 'OBJECT', 'SIMPLE', 'BITPIX',
                    'NAXIS', 'EXTEND', 'NEXTEND', 'EXTNAME', 'NAXIS1',
                    'NAXIS2', 'QUALITY']
        for kw, val in hdu0_keywords.items():
            if kw in bad_keys:
                continue
            if kw in hdu.header:
                hdu.header[kw] = val
            else:
                hdu.header.append((kw, val))
        return hdu

    def _make_target_extension(self, ext_info={}):
        """Create the 'TARGETTABLES' extension (i.e. extension #1)."""
        # Turn the data arrays into fits columns and initialize the HDU
        coldim = '({},{})'.format(self.n_cols, self.n_rows)
        eformat = '{}E'.format(self.n_rows * self.n_cols)
        jformat = '{}J'.format(self.n_rows * self.n_cols)
        cols = []
        cols.append(fits.Column(name='TIME', format='D', unit='BJD - 2454833',
                                array=self.time))
        cols.append(fits.Column(name='TIMECORR', format='E', unit='D',
                                array=self.timecorr))
        cols.append(fits.Column(name='CADENCENO', format='J',
                                array=self.cadenceno))
        cols.append(fits.Column(name='RAW_CNTS', format=jformat,
                                unit='count', dim=coldim,
                                array=self.raw_cnts))
        cols.append(fits.Column(name='FLUX', format=eformat,
                                unit='e-/s', dim=coldim,
                                array=self.flux))
        cols.append(fits.Column(name='FLUX_ERR', format=eformat,
                                unit='e-/s', dim=coldim,
                                array=self.flux_err))
        cols.append(fits.Column(name='FLUX_BKG', format=eformat,
                                unit='e-/s', dim=coldim,
                                array=self.flux_bkg))
        cols.append(fits.Column(name='FLUX_BKG_ERR', format=eformat,
                                unit='e-/s', dim=coldim,
                                array=self.flux_bkg_err))
        cols.append(fits.Column(name='COSMIC_RAYS', format=eformat,
                                unit='e-/s', dim=coldim,
                                array=self.cosmic_rays))
        cols.append(fits.Column(name='QUALITY', format='J',
                                array=self.quality))
        cols.append(fits.Column(name='POS_CORR1', format='E', unit='pixels',
                                array=self.pos_corr1))
        cols.append(fits.Column(name='POS_CORR2', format='E', unit='pixels',
                                array=self.pos_corr2))
        coldefs = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(coldefs)

        # Set the header with defaults
        template = self._header_template(1)
        for kw in template:
            if kw not in ['XTENSION', 'NAXIS1', 'NAXIS2', 'CHECKSUM', 'BITPIX']:
                try:
                    hdu.header[kw] = (self.keywords[kw],
                                      self.keywords.comments[kw])
                except KeyError:
                    hdu.header[kw] = (template[kw],
                                      template.comments[kw])
        wcs_keywords = {'CTYPE1': '1CTYP{}',
                        'CTYPE2': '2CTYP{}',
                        'CRPIX1': '1CRPX{}',
                        'CRPIX2': '2CRPX{}',
                        'CRVAL1': '1CRVL{}',
                        'CRVAL2': '2CRVL{}',
                        'CUNIT1': '1CUNI{}',
                        'CUNIT2': '2CUNI{}',
                        'CDELT1': '1CDLT{}',
                        'CDELT2': '2CDLT{}',
                        'PC1_1': '11PC{}',
                        'PC1_2': '12PC{}',
                        'PC2_1': '21PC{}',
                        'PC2_2': '22PC{}'}
        # Override defaults using data calculated in from_fits_images
        for kw in ext_info.keys():
            if kw in wcs_keywords.keys():
                for x in [4, 5, 6, 7, 8, 9]:
                    hdu.header[wcs_keywords[kw].format(x)] = ext_info[kw]
            else:
                hdu.header[kw] = ext_info[kw]
        return hdu

    def _make_aperture_extension(self):
        """Create the aperture mask extension (i.e. extension #2)."""
        mask = 3 * np.ones((self.n_rows, self.n_cols), dtype='int32')
        hdu = fits.ImageHDU(mask)

        # Set the header from the template TPF again
        template = self._header_template(2)
        for kw in template:
            if kw not in ['XTENSION', 'NAXIS1', 'NAXIS2', 'CHECKSUM', 'BITPIX']:
                try:
                    hdu.header[kw] = (self.keywords[kw],
                                      self.keywords.comments[kw])
                except KeyError:
                    hdu.header[kw] = (template[kw],
                                      template.comments[kw])

        # Override the defaults where necessary
        for keyword in ['CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CUNIT1',
                        'CUNIT2', 'CDELT1', 'CDELT2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']:
                hdu.header[keyword] = ""  # override wcs keywords
        hdu.header['EXTNAME'] = 'APERTURE'
        return hdu


class TessTargetPixelFile(TargetPixelFile):
    """
    Defines a TargetPixelFile class for the TESS Mission.
    Enables extraction of raw lightcurves and centroid positions.

    Parameters
    ----------
    path : str
        Path to a Kepler Target Pixel (FITS) File.
    quality_bitmask : str or int
        Bitmask specifying quality flags of cadences that should be ignored.
    kwargs : dict
        Keyword arguments passed to `astropy.io.fits.open()`.
    """
    def __init__(self, path, quality_bitmask='default', **kwargs):
        super(TessTargetPixelFile, self).__init__(path,
                                                  quality_bitmask=quality_bitmask,
                                                  **kwargs)
        self.quality_mask = TessQualityFlags.create_quality_mask(
                                quality_array=self.hdu[1].data['QUALITY'],
                                bitmask=quality_bitmask)
        # Early TESS releases had cadences with time=NaN (i.e. missing data)
        # which were not flagged by a QUALITY flag yet; the line below prevents
        # these cadences from being used. They would break most methods!
        self.quality_mask &= np.isfinite(self.hdu[1].data['TIME'])

        # Check the TELESCOP keyword and warn the user if it's not 'TESS'
        try:
            telescop = self.header['telescop']
            if telescop == 'Kepler':
                warnings.warn("A Kepler data product is being opened using the "
                              "`TessTargetPixelFile` class. "
                              "Please use `KeplerTargetPixelFile` instead.",
                              LightkurveWarning)
            elif telescop != 'TESS':
                warnings.warn("TessTargetPixelFile encountered 'TELESCOP' "
                              "keyword '{}' instead of 'TESS'".format(telescop),
                              LightkurveWarning)
        except KeyError:
            log.debug("TessTargetPixelFile encountered a file without 'TELESCOP' keyword.")

        # Use the TICID keyword as the default targetid
        try:
            self.targetid = self.header['TICID']
        except KeyError:
            self.targetid = None

    def __repr__(self):
        return('TessTargetPixelFile(TICID: {})'.format(self.targetid))

    @property
    def pipeline_mask(self):
        """Returns the optimal aperture mask used by the TESS pipeline.

        For details on how the mask is stored in a TPF, see Section 6 of the
        Data Products documentation (EXP-TESS-ARC-ICD-TM-0014.pdf).
        """
        return self.hdu[2].data & 2 > 0

    @property
    def background_mask(self):
        """Returns the background mask used by the TESS pipeline."""
        return self.hdu[2].data & 4 > 0

    @property
    def sector(self):
        try:
            return self.header['SECTOR']
        except KeyError:
            return None

    @property
    def camera(self):
        try:
            return self.header['CAMERA']
        except KeyError:
            return None

    @property
    def ccd(self):
        try:
            return self.header['CCD']
        except KeyError:
            return None

    @property
    def mission(self):
        return 'TESS'

    @property
    def astropy_time(self):
        """Returns an AstroPy Time object for all good-quality cadences."""
        return btjd_to_astropy_time(btjd=self.time)

    def extract_aperture_photometry(self, aperture_mask='pipeline'):
        """Performs aperture photometry.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', or 'all'
            A boolean array describing the aperture such that `False` means
            that the pixel will be masked out.
            If the string 'all' is passed, all pixels will be used.
            The default behaviour is to use the TESS pipeline mask.
        Returns
        -------
        lc : TessLightCurve object
            Contains the summed flux within the aperture for each cadence.
        """
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        if aperture_mask.sum() == 0:
            log.warning('Warning: aperture mask contains zero pixels.')
        centroid_col, centroid_row = self.estimate_centroids(aperture_mask)
        # Ignore warnings related to zero or negative errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            flux_err = np.nansum(self.flux_err[:, aperture_mask]**2, axis=1)**0.5

        keys = {'centroid_col': centroid_col,
                'centroid_row': centroid_row,
                'quality': self.quality,
                'sector': self.sector,
                'camera': self.camera,
                'ccd': self.ccd,
                'cadenceno': self.cadenceno,
                'ra': self.ra,
                'dec': self.dec,
                'label': self.header['OBJECT'],
                'targetid': self.targetid}
        return TessLightCurve(time=self.time,
                              time_format='btjd',
                              time_scale='tdb',
                              flux=np.nansum(self.flux[:, aperture_mask], axis=1),
                              flux_err=flux_err,
                              **keys)

    def get_bkg_lightcurve(self, aperture_mask=None):
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        # Ignore warnings related to zero or negative errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            flux_bkg_err = np.nansum(self.flux_bkg_err[:, aperture_mask]**2, axis=1)**0.5
        keys = {'quality': self.quality,
                'sector': self.sector,
                'camera': self.camera,
                'ccd': self.ccd,
                'cadenceno': self.cadenceno,
                'ra': self.ra,
                'dec': self.dec,
                'label': self.header['OBJECT'],
                'targetid': self.targetid}
        return TessLightCurve(time=self.time,
                              time_format='btjd',
                              time_scale='tdb',
                              flux=np.nansum(self.flux_bkg[:, aperture_mask], axis=1),
                              flux_err=flux_bkg_err,
                              **keys)
