"""Defines LightCurveFile classes, i.e. files that contain LightCurves."""

from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from astropy.io import fits as pyfits
from astropy.table import Table

from .utils import (bkjd_to_time, KeplerQualityFlags, TessQualityFlags)
from .mast import search_kepler_lightcurve_products, download_products, ArchiveError


__all__ = ['KeplerLightCurveFile', 'TessLightCurveFile']


class LightCurveFile(object):
    """Defines a generic class to handle light curve files.

    Attributes
    ----------
    path : str
        Directory path or url to a lightcurve FITS file.
    kwargs : dict
        Keyword arguments to be passed to astropy.io.fits.open.
    """
    def __init__(self, path, **kwargs):
        self.path = path
        self.hdu = pyfits.open(self.path, **kwargs)

    def header(self, ext=0):
        """Header of the object at extension `ext`"""
        return self.hdu[ext].header

    @property
    def time(self):
        """Time measurements"""
        return self.hdu[1].data['TIME'][self.quality_mask]

    @property
    def timeobj(self):
        """Returns the human-readable date for all good-quality cadences."""
        return bkjd_to_time(bkjd=self.time,
                            timecorr=self.hdu[1].data['TIMECORR'][self.quality_mask],
                            timslice=self.hdu[1].header['TIMSLICE'])

    @property
    def SAP_FLUX(self):
        """Returns a LightCurve object for SAP_FLUX"""
        return self.get_lightcurve('SAP_FLUX')

    @property
    def PDCSAP_FLUX(self):
        """Returns a LightCurve object for PDCSAP_FLUX"""
        return self.get_lightcurve('PDCSAP_FLUX')

    @property
    def cadenceno(self):
        """Cadence number"""
        return self.hdu[1].data['CADENCENO'][self.quality_mask]

    def _flux_types(self):
        """Returns a list of available flux types for this light curve file"""
        types = [n for n in self.hdu[1].data.columns.names if 'FLUX' in n]
        types = [n for n in types if not ('ERR' in n)]
        return types


class KeplerLightCurveFile(LightCurveFile):
    """Defines a class for a given light curve FITS file from NASA's Kepler and
    K2 missions.

    Attributes
    ----------
    path : str
        Directory path or url to a lightcurve FITS file.
    quality_bitmask : str or int
        Bitmask specifying quality flags of cadences that should be ignored.
        If `None` is passed, then no cadences are ignored.
        If a string is passed, it has the following meaning:

            * default: recommended quality mask
            * hard: removes more flags, known to remove good data
            * hardest: removes all data that has been flagged

        See the `KeplerQualityFlags` class for details on the bitmasks.
    kwargs : dict
        Keyword arguments to be passed to astropy.io.fits.open.
    """

    def __init__(self, path, quality_bitmask='default', **kwargs):
        super(KeplerLightCurveFile, self).__init__(path, **kwargs)
        self.quality_bitmask = quality_bitmask
        self.quality_mask = self._quality_mask(quality_bitmask)

    @staticmethod
    def from_archive(target, cadence='long', quarter=None, month=None,
                     campaign=None, **kwargs):
        """Fetch a Light Curve File from the Kepler/K2 data archive at MAST.

        Raises an `ArchiveError` if a unique file cannot be found.  For example,
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
            lightcurve files for each quarter, each covering one month.
            Hence, if cadence='short' you need to specify month=1, 2, or 3.
        kwargs : dict
            Keywords arguments passed to `KeplerLightCurveFile`.

        Returns
        -------
        tpf : KeplerLightCurveFile object.
        """
        products = search_kepler_lightcurve_products(target=target, cadence=cadence,
                                                     quarter=quarter, campaign=campaign)
        if cadence == 'short' and len(products) > 1:
            if month is None:
                raise ArchiveError("Found {} different lightcurve files "
                                   "for target {} in Quarter {}."
                                   "Please specify the month (1, 2, or 3)."
                                   "".format(len(products), target, quarter))
            products = Table(products[month+1])
        elif len(products) > 1:
            raise ArchiveError("Found {} different lightcurve files "
                               "for target {}. Please specify quarter/month "
                               "or campaign number."
                               "".format(len(products), target))
        elif len(products) < 1:
            raise ArchiveError("No lightcurve file found for {} at MAST.".format(target))
        path = download_products(products)[0]
        return KeplerLightCurveFile(path, **kwargs)

    def __repr__(self):
        if self.mission is None:
            return('KeplerLightCurveFile(ID: {})'.format(self.keplerid))
        elif self.mission.lower() == 'kepler':
            return('KeplerLightCurveFile(KIC: {})'.format(self.keplerid))
        elif self.mission.lower() == 'k2':
            return('KeplerLightCurveFile(EPIC: {})'.format(self.keplerid))

    def _quality_mask(self, bitmask):
        """Returns a boolean mask which flags all good-quality cadences.

        Parameters
        ----------
        bitmask : str or int
            Bitmask. See ref. [1], table 2-3.

        Returns
        -------
        boolean_mask : array of bool
            Boolean array in which `True` means the data is of good quality.
        """
        if bitmask is None:
            return np.ones(len(self.hdu[1].data['TIME']), dtype=bool)

        if isinstance(bitmask, str):
            bitmask = KeplerQualityFlags.OPTIONS[bitmask]
        return (self.hdu[1].data['SAP_QUALITY'] & bitmask) == 0

    def get_lightcurve(self, flux_type, centroid_type='MOM_CENTR'):
        if flux_type in self._flux_types():
            # We did not import lightcurve at the top to prevent circular imports
            from .lightcurve import KeplerLightCurve
            return KeplerLightCurve(
                        self.hdu[1].data['TIME'][self.quality_mask],
                        self.hdu[1].data[flux_type][self.quality_mask],
                        flux_err=self.hdu[1].data[flux_type + "_ERR"][self.quality_mask],
                        centroid_col=self.hdu[1].data[centroid_type + "1"][self.quality_mask],
                        centroid_row=self.hdu[1].data[centroid_type + "2"][self.quality_mask],
                        quality=self.hdu[1].data['SAP_QUALITY'][self.quality_mask],
                        quality_bitmask=self.quality_bitmask,
                        channel=self.channel,
                        campaign=self.campaign,
                        quarter=self.quarter,
                        mission=self.mission,
                        cadenceno=self.cadenceno,
                        keplerid=self.keplerid)
        else:
            raise KeyError("{} is not a valid flux type. Available types are: {}".
                           format(flux_type, self._flux_types))

    @property
    def channel(self):
        """Channel number"""
        return self.header(ext=0)['CHANNEL']

    @property
    def keplerid(self):
        return self.header(ext=0)['KEPLERID']

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

    def compute_cotrended_lightcurve(self, cbvs=[1, 2], **kwargs):
        """Returns a LightCurve object after cotrending the SAP_FLUX
        against the cotrending basis vectors.

        Parameters
        ----------
        cbvs : list of ints
            The list of cotrending basis vectors to fit to the data. For example,
            [1, 2] will fit the first two basis vectors.
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

    def plot(self, flux_types=None, style='fast', **kwargs):
        """Plot all the light curves contained in this light curve file.

        Parameters
        ----------
        flux_types : str or list of str
            List of flux types to plot. Default is to plot all available.
            (For Kepler the default fluxes are 'SAP_FLUX' and 'PDCSAP-FLUX'.
        style : str
            matplotlib.pyplot.style.context, default is 'fast'
        kwargs : dict
            Dictionary of keyword arguments to be passed to
            `KeplerLightCurve.plot()`.
        """
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
                lc.plot(label=ft, **kwargs)


class TessLightCurveFile(LightCurveFile):
    """Defines a class for a given light curve FITS file from NASA's TESS
    mission.

    Attributes
    ----------
    path : str
        Directory path or url to a lightcurve FITS file.
    quality_bitmask : str or int
        Bitmask specifying quality flags of cadences that should be ignored.
        If a string is passed, it has the following meaning:

            * default: recommended quality mask
            * hard: removes more flags, known to remove good data
            * hardest: removes all data that has been flagged

        See the `TessQualityFlags` class for details on the bitmasks.
    kwargs : dict
        Keyword arguments to be passed to astropy.io.fits.open.
    """

    def __init__(self, path, quality_bitmask='default', **kwargs):
        super(TessLightCurveFile, self).__init__(path, **kwargs)
        self.quality_bitmask = quality_bitmask
        self.quality_mask = self._quality_mask(quality_bitmask)

    def __repr__(self):
        return('TessLightCurveFile(TICID: {})'.format(self.ticid))

    def _quality_mask(self, bitmask):
        """Returns a boolean mask which flags all good-quality cadences.

        Parameters
        ----------
        bitmask : str or int
            Bitmask. See ref. [1], table 2-3.

        Returns
        -------
        boolean_mask : array of bool
            Boolean array in which `True` means the data is of good quality.
        """
        if bitmask is None:
            return np.ones(len(self.hdu[1].data['TIME']), dtype=bool)

        if isinstance(bitmask, str):
            bitmask = TessQualityFlags.OPTIONS[bitmask]
        return (self.hdu[1].data['QUALITY'] & bitmask) == 0

    @property
    def ticid(self):
        return self.header(ext=0)['TICID']

    def get_lightcurve(self, flux_type, centroid_type='MOM_CENTR'):
        if flux_type in self._flux_types():
            # We did not import lightcurve at the top to prevent circular imports
            from .lightcurve import TessLightCurve
            return TessLightCurve(
                        self.hdu[1].data['TIME'][self.quality_mask],
                        self.hdu[1].data[flux_type][self.quality_mask],
                        flux_err=self.hdu[1].data[flux_type + "_ERR"][self.quality_mask],
                        centroid_col=self.hdu[1].data[centroid_type + "1"][self.quality_mask],
                        centroid_row=self.hdu[1].data[centroid_type + "2"][self.quality_mask],
                        quality=self.hdu[1].data['QUALITY'][self.quality_mask],
                        quality_bitmask=self.quality_bitmask,
                        cadenceno=self.cadenceno,
                        ticid=self.ticid)
