import warnings

from astropy.io import fits
from astropy.table import Table
from matplotlib import patches
import numpy as np

from .lightcurve import KeplerLightCurve, LightCurve
from .prf import SimpleKeplerPRF
from .utils import KeplerQualityFlags, TessQualityFlags, plot_image
from .mast import search_kepler_tpf_products, download_products, ArchiveError


__all__ = ['KeplerTargetPixelFile', 'TessTargetPixelFile']


class TargetPixelFile(object):
    """
    TargetPixelFile class
    """
    def __init__(self, path, **kwargs):
        self.path = path
        self.hdu = fits.open(self.path, **kwargs)

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
    def n_good_cadences(self):
        """Returns the number of good-quality cadences."""
        return self.quality_mask.sum()

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

    def header(self, ext=0):
        """Returns the header for a given extension."""
        return self.hdu[ext].header

    @property
    def ra(self):
        try:
            return self.header()['RA_OBJ']
        except KeyError:
            return None

    @property
    def dec(self):
        try:
            return self.header()['DEC_OBJ']
        except KeyError:
            return None

    def get_bkg_lightcurve(self, aperture_mask=None):
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        return LightCurve(flux=np.nansum(self.flux_bkg[:, aperture_mask], axis=1),
                          time=self.time, flux_err=self.flux_bkg_err)

    def _parse_aperture_mask(self, aperture_mask):
        """Parse the `aperture_mask` parameter as given by a user.

        The `aperture_mask` parameter is accepted by a number of methods.
        This method ensures that the parameter is always parsed in the same way.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', 'all', or None
            A boolean array describing the aperture such that `False` means
            that the pixel will be masked out.
            If None or 'all' are passed, a mask that is `True` everywhere will
            be returned.
            If 'pipeline' is passed, the mask suggested by the Kepler pipeline
            will be returned.

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
        self._last_aperture_mask = aperture_mask
        return aperture_mask

    def to_lightcurve(self):
        """Returns a raw light curve of the TPF.

        Returns
        -------
        lc : LightCurve object
            Array containing the summed or detrended flux within the aperture
            for each cadence.
        """
        pass


class TessTargetPixelFile(TargetPixelFile):
    """
    Defines a TargetPixelFile class for the TESS Mission.
    Enables extraction of raw lightcurves and centroid positions.

    Attributes
    ----------
    path : str
        Path to a Kepler Target Pixel (FITS) File.
    quality_bitmask : str or int
        Bitmask specifying quality flags of cadences that should be ignored.
        If a string is passed, it has the following meaning:

            * "default": recommended quality mask
            * "hard": removes more flags, known to remove good data
            * "hardest": removes all data that has been flagged

    References
    ----------
    """
    def __init__(self, path, quality_bitmask=TessQualityFlags.DEFAULT_BITMASK,
                 **kwargs):
        super(TessTargetPixelFile, self).__init__(path, **kwargs)
        self.quality_bitmask = quality_bitmask
        self.quality_mask = self._quality_mask(quality_bitmask)

    def __repr__(self):
        return('TessTargetPixelFile(TICID: {})'.format(self.ticid))

    def _quality_mask(self, bitmask):
        """Returns a boolean mask which flags all good-quality cadences.

        Parameters
        ----------
        bitmask : str or int
            Bitmask. See ref. [1], table 2-3.
        """
        if bitmask is None:
            return np.ones(len(self.hdu[1].data['TIME']), dtype=bool)
        elif isinstance(bitmask, str):
            bitmask = TessQualityFlags.OPTIONS[bitmask]
        return (self.hdu[1].data['QUALITY'] & bitmask) == 0

    def to_lightcurve(self, aperture_mask='pipeline'):
        """Performs aperture photometry.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', or 'all'
            A boolean array describing the aperture such that `False` means
            that the pixel will be masked out.
            If the string 'all' is passed, all pixels will be used.
            The default behaviour is to use the Kepler pipeline mask.

        Returns
        -------
        lc : KeplerLightCurve object
            Array containing the summed flux within the aperture for each
            cadence.
        """
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        centroid_col, centroid_row = self.centroids(aperture_mask)

        return TessLightCurve(flux=np.nansum(self.flux[:, aperture_mask], axis=1),
                              time=self.time,
                              flux_err=np.nansum(self.flux_err[:, aperture_mask]**2, axis=1)**0.5,
                              centroid_col=centroid_col,
                              centroid_row=centroid_row,
                              quality=self.quality,
                              mission=self.mission,
                              cadenceno=self.cadenceno)


class KeplerTargetPixelFile(TargetPixelFile):
    """
    Defines a TargetPixelFile class for the Kepler/K2 Mission.
    Enables extraction of raw lightcurves and centroid positions.

    Attributes
    ----------
    path : str
        Path to a Kepler Target Pixel (FITS) File.
    quality_bitmask : str or int
        Bitmask specifying quality flags of cadences that should be ignored.
        If a string is passed, it has the following meaning:

            * "default": recommended quality mask
            * "hard": removes more flags, known to remove good data
            * "hardest": removes all data that has been flagged

    References
    ----------
    .. [1] Kepler: A Search for Terrestrial Planets. Kepler Archive Manual.
        http://archive.stsci.edu/kepler/manuals/archive_manual.pdf
    """

    @staticmethod
    def from_archive(target, cadence='long', quarter=None, month=None, campaign=None):
        """Fetch a Target Pixel File from the Kepler/K2 data archive at MAST.

        Raises an `ArchiveError` if a unique TPF cannot be found.  For example,
        this is the case if a target was observed in multiple Quarters and the
        quarter parameter is unspecified.

        Parameters
        ----------
        target : str or int
            KIC/EPIC ID or object name.
        cadence : str
            'long' or 'short'.
        quarter, campaign : int
            Kepler Quarter or K2 Campaign number.
        month : 1, 2, or 3
            For Kepler's prime mission, there are three short-cadence
            Target Pixel Files for each quarter, each covering one month.
            Hence, if cadence='short' you need to specify month=1, 2, or 3.

        Returns
        -------
        tpf : KeplerTargetPixelFile object.
        """
        products = search_kepler_tpf_products(target=target, cadence=cadence,
                                              quarter=quarter, campaign=campaign)
        if cadence == 'short' and len(products) > 1:
            if month is None:
                raise ArchiveError("Found {} different Target Pixel Files "
                                   "for target {} in Quarter {}."
                                   "Please specify the month (1, 2, or 3)."
                                   "".format(len(products), target, quarter))
            products = Table(products[month+1])
        elif len(products) > 1:
            raise ArchiveError("Found {} different Target Pixel Files "
                               "for target {}. Please specify quarter/month "
                               "or campaign number."
                               "".format(len(products), target))
        elif len(products) < 1:
            raise ArchiveError("No Target Pixel File found for {} at MAST.".format(target))
        path = download_products(products)[0]
        return KeplerTargetPixelFile(path)

    def __repr__(self):
        return('KeplerTargetPixelFile(KEPLERID: {})'.format(self.keplerid))

    def _quality_mask(self, bitmask):
        """Returns a boolean mask which flags all good-quality cadences.

        Parameters
        ----------
        bitmask : str or int
            Bitmask. See ref. [1], table 2-3.
        """
        if bitmask is None:
            return np.ones(len(self.hdu[1].data['TIME']), dtype=bool)
        elif isinstance(bitmask, str):
            bitmask = KeplerQualityFlags.OPTIONS[bitmask]
        return (self.hdu[1].data['QUALITY'] & bitmask) == 0

    def get_prf_model(self):
        """Returns an object of SimpleKeplerPRF initialized using the
        necessary metadata in the tpf object.

        Returns
        -------
        prf : instance of SimpleKeplerPRF
        """

        return SimpleKeplerPRF(channel=self.channel, shape=self.shape[1:],
                               column=self.column, row=self.row)

    @property
    def keplerid(self):
        return self.header()['KEPLERID']

    @property
    def module(self):
        return self.header()['MODULE']

    @property
    def channel(self):
        return self.header()['CHANNEL']

    @property
    def output(self):
        return self.header()['OUTPUT']

    @property
    def column(self):
        return self.hdu['TARGETTABLES'].header['1CRV5P']

    @property
    def row(self):
        return self.hdu['TARGETTABLES'].header['2CRV5P']

    @property
    def pipeline_mask(self):
        """Returns the aperture mask used by the Kepler pipeline"""
        return self.hdu[-1].data > 2

    @property
    def quarter(self):
        """Quarter number"""
        try:
            return self.header(ext=0)['QUARTER']
        except KeyError:
            return None

    @property
    def campaign(self):
        """Campaign number"""
        try:
            return self.header(ext=0)['CAMPAIGN']
        except KeyError:
            return None

    @property
    def mission(self):
        """Mission name"""
        return self.header(ext=0)['MISSION']

    def to_fits(self):
        """Save the TPF to fits"""
        raise NotImplementedError

    def to_lightcurve(self, aperture_mask='pipeline'):
        """Performs aperture photometry.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', or 'all'
            A boolean array describing the aperture such that `False` means
            that the pixel will be masked out.
            If the string 'all' is passed, all pixels will be used.
            The default behaviour is to use the Kepler pipeline mask.

        Returns
        -------
        lc : KeplerLightCurve object
            Array containing the summed flux within the aperture for each
            cadence.
        """
        aperture_mask = self._parse_aperture_mask(aperture_mask)
        centroid_col, centroid_row = self.centroids(aperture_mask)

        return KeplerLightCurve(flux=np.nansum(self.flux[:, aperture_mask], axis=1),
                                time=self.time,
                                flux_err=np.nansum(self.flux_err[:, aperture_mask]**2, axis=1)**0.5,
                                centroid_col=centroid_col,
                                centroid_row=centroid_row,
                                quality=self.quality,
                                channel=self.channel,
                                campaign=self.campaign,
                                quarter=self.quarter,
                                mission=self.mission,
                                cadenceno=self.cadenceno)

    def centroids(self, aperture_mask='pipeline'):
        """Returns centroids based on sample moments.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', or 'all'
            A boolean array describing the aperture such that `False` means
            that the pixel will be masked out.
            If the string 'all' is passed, all pixels will be used.
            The default behaviour is to use the Kepler pipeline mask.

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
        col_centr = np.nansum(xx * aperture_mask * self.flux, axis=(1, 2)) / total_flux
        row_centr = np.nansum(yy * aperture_mask * self.flux, axis=(1, 2)) / total_flux

        return col_centr, row_centr

    def plot(self, ax=None, frame=0, cadenceno=None, bkg=False, aperture_mask=None,
            show_colorbar=True, mask_color='pink', **kwargs):
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
        aperture_mask : ndarray
            Highlight pixels selected by aperture_mask.
        show_colorbar : bool
            Whether or not to show the colorbar
        mask_color : str
            Color to show the aperture mask
        kwargs : dict
            Keywords arguments passed to `lightkurve.utils.plot_image`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if cadenceno is not None:
            try:
                frame = np.argwhere(cadenceno == self.cadenceno)[0][0]
            except IndexError:
                raise ValueError("cadenceno {} is out of bounds, "
                                 "must be in the range {}-{}.".format(
                                    cadenceno, self.cadenceno[0], self.cadenceno[-1]))
        try:
            if bkg:
                pflux = self.flux[frame] + self.flux_bkg[frame]
            else:
                pflux = self.flux[frame]
        except IndexError:
            raise ValueError("frame {} is out of bounds, must be in the range "
                             "0-{}.".format(frame, self.flux.shape[0]))
        ax = plot_image(pflux, ax=ax, title='Kepler ID: {}'.format(self.keplerid),
                extent=(self.column, self.column + self.shape[2], self.row,
                self.row + self.shape[1]), show_colorbar=show_colorbar, **kwargs)

        if aperture_mask is not None:
            aperture_mask = self._parse_aperture_mask(aperture_mask)
            for i in range(self.shape[1]):
                for j in range(self.shape[2]):
                    if aperture_mask[i, j]:
                        ax.add_patch(patches.Rectangle((j+self.column, i+self.row),
                                                       1, 1, color=mask_color, fill=True,
                                                       alpha=.6))
        return ax
