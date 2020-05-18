"""Defines LightCurveFile classes, i.e. files that contain LightCurves."""

from __future__ import division, print_function

import logging
import warnings

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.io import fits as pyfits
from astropy.io.fits import Undefined

from .utils import (bkjd_to_astropy_time, KeplerQualityFlags, TessQualityFlags,
                    LightkurveWarning, detect_filetype)

from . import MPLSTYLE

__all__ = ['LightCurveFile', 'KeplerLightCurveFile', 'TessLightCurveFile']

log = logging.getLogger(__name__)


class LightCurveFile(object):
    """Generic class to represent FITS files which contain one or more light curves.

    Parameters
    ----------
    path : str or `astropy.io.fits.HDUList` object
        Local path or remote url of a lightcurve FITS file.
        Also accepts a FITS file object already opened using AstroPy.
    kwargs : dict
        Keyword arguments to be passed to astropy.io.fits.open.
    """
    def __init__(self, path, **kwargs):
        if isinstance(path, pyfits.HDUList):
            self.path = None
            self.hdu = path
        else:
            self.path = path
            self.hdu = pyfits.open(self.path, **kwargs)

    def header(self, ext=0):
        """DEPRECATED. Please use ``get_header()`` instead."""
        warnings.warn("`LightCurveFile.header` is deprecated, please use "
                      "`LightCurveFile.get_header()` instead.",
                      LightkurveWarning)
        return self.hdu[ext].header

    def get_header(self, ext=0):
        """Returns the metadata embedded in the file.

        Light Curve Files contain embedded metadata headers spread across three
        different FITS extensions:

        1. The "PRIMARY" extension (``ext=0``) provides a metadata header
           providing details on the target and its CCD position.
        2. The "LIGHTCURVE" extension (``ext=1``) provides details on the
           data columns and the systematics removal.
        3. The "APERTURE" extension (``ext=2``) provides details on the
           aperture pixel mask and the expected coordinate system (WCS).

        Parameters
        ----------
        ext : int or str
            FITS extension name or number.

        Returns
        -------
        header : `~astropy.io.fits.header.Header`
            Header object containing metadata keywords.
        """
        return self.hdu[ext].header

    def get_keyword(self, keyword, hdu=0, default=None):
        """Returns a header keyword value.

        If the keyword is Undefined or does not exist,
        then return ``default`` instead.
        """
        try:
            kw = self.hdu[hdu].header[keyword]
        except KeyError:
            return default
        if isinstance(kw, Undefined):
            return default
        return kw

    @property
    def time(self):
        """The file's `TIME` column."""
        return self.hdu[1].data['TIME'][self.quality_mask]

    @property
    def cadenceno(self):
        """The file's `CADENCENO` column."""
        return self.hdu[1].data['CADENCENO'][self.quality_mask]

    @property
    def ra(self):
        """Right Ascension as recorded in the header's `RA_OBJ` keyword."""
        return self.get_keyword('RA_OBJ')

    @property
    def dec(self):
        """Declination as recorded in the header's `DEC_OBJ` keyword."""
        return self.get_keyword('DEC_OBJ')

    @property
    def FLUX(self):
        """Returns a `~lightkurve.lightcurve.LightCurve` object based on the
        contents of the `FLUX` column in the file, if that column exists."""
        return self.get_lightcurve('FLUX')

    @property
    def SAP_FLUX(self):
        """Returns a `~lightkurve.lightcurve.LightCurve` object based on the
        contents of the `SAP_FLUX` column in the file, if that column exists."""
        return self.get_lightcurve('SAP_FLUX')

    @property
    def PDCSAP_FLUX(self):
        """Returns a `~lightkurve.lightcurve.LightCurve` object based on the
        contents of the `PDCSAP_FLUX` column in the file, if that column exists.
        """
        return self.get_lightcurve('PDCSAP_FLUX')

    def _flux_types(self):
        """Returns a list of available flux types for this light curve file"""
        types = [n for n in self.hdu[1].data.columns.names if 'FLUX' in n]
        types = [n for n in types if not ('ERR' in n)]
        return types

    def _get_quality(self):
        """Returns the quality flag vector, which may go by different names
        """
        if 'QUALITY' in self.hdu[1].data.columns.names:
            quality_vector = self.hdu[1].data['QUALITY']
        elif 'SAP_QUALITY' in self.hdu[1].data.columns.names:
            quality_vector = self.hdu[1].data['SAP_QUALITY']
        else:
            quality_vector = np.zeros(len(self.hdu[1].data['TIME']))
        return quality_vector

    def _create_plot(self, method='plot', flux_types=None, style='lightkurve',
                     **kwargs):
        """Implements `plot()`, `scatter()`, and `errorbar()` to avoid code duplication.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib Axes object.
        """
        if style is None or style == 'lightkurve':
            style = MPLSTYLE
        with plt.style.context(style):
            if not ('ax' in kwargs):
                fig, ax = plt.subplots(1)
                kwargs['ax'] = ax
            if flux_types is None:
                flux_types = self._flux_types()
            if isinstance(flux_types, str):
                flux_types = [flux_types]
            for idx, ft in enumerate(flux_types):
                lc = self.get_lightcurve(ft)
                kwargs['color'] = np.asarray(mpl.rcParams['axes.prop_cycle'])[idx]['color']
                if method == 'plot':
                    lc.plot(label=ft, **kwargs)
                elif method == 'scatter':
                    lc.scatter(label=ft, **kwargs)
                elif method == 'errorbar':
                    lc.errorbar(label=ft, **kwargs)


    def plot(self, flux_types=None, style='lightkurve', **kwargs):
        """Plot the light curve file using matplotlib's `plot` method.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        flux_types : list or None
            Which fluxes in the LCF to plot. Default is lcf._flux_types().
            For Kepler this is PDCSAP and SAP flux. Pass a list to change flux
            types.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        return self._create_plot(method='plot', flux_types=flux_types,
                                 style=style, **kwargs)


    def scatter(self, flux_types=None, style='lightkurve', **kwargs):
        """Plot the light curve file using matplotlib's `scatter` method.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        flux_types : list or None
            Which fluxes in the LCF to plot. Default is lcf._flux_types().
            For Kepler this is PDCSAP and SAP flux. Pass a list to change flux
            types.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        return self._create_plot(method='scatter', flux_types=flux_types,
                                 style=style, **kwargs)

    def errorbar(self, flux_types=None, style='lightkurve', **kwargs):
        """Plot the light curve file using matplotlib's `errorbar` method.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        flux_types : list or None
            Which fluxes in the LCF to plot. Default is lcf._flux_types().
            For Kepler this is PDCSAP and SAP flux. Pass a list to change flux
            types.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        return self._create_plot(method='errorbar', flux_types=flux_types,
                                 style=style, **kwargs)


class KeplerLightCurveFile(LightCurveFile):
    """Subclass of :class:`LightCurveFile <lightkurve.lightcurvefile.LightCurveFile>`
    to represent files generated by NASA's Kepler pipeline.

    Parameters
    ----------
    path : str
        Local path or remote url of a FITS file in Kepler's lightcurve format.
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
        Keyword arguments to be passed to astropy.io.fits.open.
    """
    def __init__(self, path, quality_bitmask='default', **kwargs):
        super(KeplerLightCurveFile, self).__init__(path, **kwargs)

        # check to make sure the correct filetype has been provided
        filetype = detect_filetype(self.header())
        if filetype == 'TessLightCurveFile':
            warnings.warn("A TESS data product is being opened using the "
                          "`KeplerLightCurveFile` class. "
                          "Please use `TessLightCurveFile` instead.",
                          LightkurveWarning)
        elif filetype is None:
            warnings.warn("Given fits file not recognized as Kepler or TESS "
                          "observation.", LightkurveWarning)
        elif "TargetPixelFile" in filetype:
            warnings.warn("A `TargetPixelFile` object is being opened as a "
                          "`KeplerLightCurveFile`. "
                          "Please use `KeplerTargetPixelFile` instead.",
                          LightkurveWarning)

        self.quality_bitmask = quality_bitmask
        self.quality_mask = KeplerQualityFlags.create_quality_mask(
                                quality_array=self.hdu[1].data['SAP_QUALITY'],
                                bitmask=quality_bitmask)
        self.targetid = self.get_keyword('KEPLERID')

    def __repr__(self):
        return('KeplerLightCurveFile(ID: {})'.format(self.targetid))

    @property
    def astropy_time(self):
        """Returns an AstroPy Time object for all good-quality cadences."""
        return bkjd_to_astropy_time(bkjd=self.time)

    def get_lightcurve(self, flux_type, centroid_type='MOM_CENTR'):
        if centroid_type+"1" in self.hdu[1].data.columns.names:
            centroid_col = self.hdu[1].data[centroid_type + "1"][self.quality_mask]
            centroid_row = self.hdu[1].data[centroid_type + "2"][self.quality_mask]
        else:
            centroid_col = np.repeat(np.NaN, self.quality_mask.sum())
            centroid_row = np.repeat(np.NaN, self.quality_mask.sum())
        if flux_type in self._flux_types():
            # We did not import lightcurve at the top to prevent circular imports
            from .lightcurve import KeplerLightCurve

            f = self.hdu[1].data[flux_type][self.quality_mask]
            fe = self.hdu[1].data[flux_type + "_ERR"][self.quality_mask]

            if flux_type == 'SAP_FLUX':
                f /= self.hdu[1].header.get('FLFRCSAP', 1)
                fe /= self.hdu[1].header.get('FLFRCSAP', 1)
                f /= self.hdu[1].header.get('CROWDSAP', 1)
                fe /= self.hdu[1].header.get('CROWDSAP', 1)

            return KeplerLightCurve(
                time=self.hdu[1].data['TIME'][self.quality_mask],
                time_format='bkjd',
                time_scale='tdb',
                flux=f,
                flux_err=fe,
                centroid_col=centroid_col,
                centroid_row=centroid_row,
                quality=self._get_quality()[self.quality_mask],
                quality_bitmask=self.quality_bitmask,
                channel=self.channel,
                campaign=self.campaign,
                quarter=self.quarter,
                mission=self.mission,
                cadenceno=self.cadenceno,
                targetid=self.targetid,
                label=self.get_keyword('OBJECT'),
                ra=self.ra,
                dec=self.dec)
        else:
            raise KeyError("{} is not a valid flux type. Available types are: {}".
                           format(flux_type, self._flux_types()))

    @property
    def channel(self):
        """Kepler CCD channel number. ('CHANNEL' header keyword)"""
        return self.get_keyword('CHANNEL')

    @property
    def obsmode(self):
        """'short cadence' or 'long cadence'. ('OBSMODE' header keyword)"""
        return self.get_keyword('OBSMODE')

    @property
    def pos_corr1(self):
        """Returns the column position correction."""
        return self.hdu[1].data['POS_CORR1'][self.quality_mask]

    @property
    def pos_corr2(self):
        """Returns the row position correction."""
        return self.hdu[1].data['POS_CORR2'][self.quality_mask]

    @property
    def quarter(self):
        """Kepler quarter number. ('QUARTER' header keyword)"""
        return self.get_keyword('QUARTER')

    @property
    def campaign(self):
        """K2 Campaign number. ('CAMPAIGN' header keyword)"""
        return self.get_keyword('CAMPAIGN')

    @property
    def mission(self):
        """'Kepler' or 'K2'. ('MISSION' header keyword)"""
        return self.get_keyword('MISSION')

    def compute_cotrended_lightcurve(self, cbvs=(1, 2), **kwargs):
        """Returns a LightCurve object after cotrending the SAP_FLUX
        against the cotrending basis vectors.

        Parameters
        ----------
        cbvs : tuple or list of ints
            The list of cotrending basis vectors to fit to the data. For example,
            (1, 2) will fit the first two basis vectors.
        kwargs : dict
            Dictionary of keyword arguments to be passed to
            KeplerCBVCorrector.correct.

        Returns
        -------
        lc : LightCurve object
            CBV flux-corrected lightcurve.
        """
        from .correctors import KeplerCBVCorrector
        return KeplerCBVCorrector(self).correct(cbvs=cbvs, **kwargs)


class TessLightCurveFile(LightCurveFile):
    """Subclass of :class:`LightCurveFile <lightkurve.lightcurvefile.LightCurveFile>`
    to represent files generated by NASA's TESS pipeline.

    Parameters
    ----------
    path : str
        Local path or remote url of a FITS file in TESS's lightcurve format.
    quality_bitmask : str or int
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. If a string is passed, it has the
        following meaning:

            * "none": no cadences will be ignored (`quality_bitmask=0`).
            * "default": cadences with severe quality issues will be ignored
              (`quality_bitmask=175`).
            * "hard": more conservative choice of flags to ignore
              (`quality_bitmask=7407`). This is known to remove good data.
            * "hardest": removes all data that has been flagged
              (`quality_bitmask=8191`). This mask is not recommended.

        See the :class:`TessQualityFlags` class for details on the bitmasks.
    kwargs : dict
        Keyword arguments to be passed to astropy.io.fits.open.
    """
    def __init__(self, path, quality_bitmask='default', **kwargs):
        super(TessLightCurveFile, self).__init__(path, **kwargs)

        # check to make sure the correct filetype has been provided
        filetype = detect_filetype(self.header())
        if filetype == 'KeplerLightCurveFile':
            warnings.warn("A Kepler data product is being opened using the "
                          "`TessLightCurveFile` class. "
                          "Please use `KeplerLightCurveFile` instead.",
                          LightkurveWarning)
        elif filetype is None:
            warnings.warn("Given fits file not recognized as Kepler or TESS "
                          "observation.", LightkurveWarning)
        elif "TargetPixelFile" in filetype:
            warnings.warn("A `TargetPixelFile` object is being opened as a "
                          "`TessLightCurveFile`. "
                          "Please use `TessTargetPixelFile` instead.",
                          LightkurveWarning)

        self.quality_bitmask = quality_bitmask
        self.quality_mask = TessQualityFlags.create_quality_mask(
                        quality_array=self._get_quality(),
                        bitmask=quality_bitmask)

        # Early TESS releases had cadences with time=NaN (i.e. missing data)
        # which were not flagged by a QUALITY flag yet; the line below prevents
        # these cadences from being used. They would break most methods!
        self.quality_mask &= np.isfinite(self.hdu[1].data['TIME'])
        self.targetid = self.get_keyword('TICID')

    def __repr__(self):
        return('TessLightCurveFile(TICID: {})'.format(self.targetid))

    @property
    def sector(self):
        """TESS Sector number ('SECTOR' header keyword)."""
        return self.get_keyword('SECTOR')

    @property
    def camera(self):
        """TESS Camera number ('CAMERA' header keyword)."""
        return self.get_keyword('CAMERA')

    @property
    def ccd(self):
        """TESS CCD number ('CCD' header keyword)."""
        return self.get_keyword('CCD')

    @property
    def mission(self):
        return 'TESS'

    def get_lightcurve(self, flux_type, centroid_type='MOM_CENTR'):
        if centroid_type+"1" in self.hdu[1].data.columns.names:
            centroid_col = self.hdu[1].data[centroid_type + "1"][self.quality_mask]
            centroid_row = self.hdu[1].data[centroid_type + "2"][self.quality_mask]
        else:
            centroid_col = np.repeat(np.NaN, self.quality_mask.sum())
            centroid_row = np.repeat(np.NaN, self.quality_mask.sum())

        if flux_type in self._flux_types():
            # We did not import TessLightCurve at the top to prevent circular imports
            from .lightcurve import TessLightCurve
            return TessLightCurve(
                time=self.hdu[1].data['TIME'][self.quality_mask],
                time_format='btjd',
                time_scale='tdb',
                flux=self.hdu[1].data[flux_type][self.quality_mask],
                flux_err=self.hdu[1].data[flux_type + "_ERR"][self.quality_mask],
                centroid_col=centroid_col,
                centroid_row=centroid_row,
                quality=self._get_quality()[self.quality_mask],
                quality_bitmask=self.quality_bitmask,
                cadenceno=self.cadenceno,
                targetid=self.targetid,
                label=self.get_keyword('OBJECT'),
                sector=self.sector,
                camera=self.camera,
                ccd=self.ccd,
                ra=self.ra,
                dec=self.dec)
        else:
            raise KeyError("{} is not a valid flux type. Available types are: {}".
                           format(flux_type, self._flux_types()))
