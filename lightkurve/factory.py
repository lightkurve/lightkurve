"""Defines TargetPixelFileFactory."""
from __future__ import division
import datetime
import os
import warnings
import logging

from astropy.io import fits

import numpy as np

from . import PACKAGEDIR
from .utils import LightkurveWarning
from .targetpixelfile import TargetPixelFile, KeplerTargetPixelFile, TessTargetPixelFile

log = logging.getLogger(__name__)

class FactoryError(Exception):
    """Raised if there is a problem creating a TPF."""
    pass


class TargetPixelFileFactory(object):
    """Class to create a KeplerTargetPixelFile."""

    def __init__(self, n_cadences, n_rows, n_cols, targetid="unnamed-target",
                 keywords=None):
        self.n_cadences = n_cadences
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.targetid = targetid
        if keywords is None:
            self.keywords = {}
        else:
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

    def add_cadence(self, frameno, raw_cnts=None, flux=None, flux_err=None,
                    flux_bkg=None, flux_bkg_err=None, cosmic_rays=None,
                    header=None):
        """Populate the data for a single cadence."""
        if frameno >= self.n_cadences:
            raise FactoryError('Can not add cadence {}, n_cadences set to {}'.format(frameno, self.n_cadences))
        if header is None:
            header = {}

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
        elif 'BMJD_OBS' in header and 'FRAMTIME' in header:
            self.time[frameno] = header['BMJD_OBS'] + header['FRAMTIME']/2./86400.0
        if 'TIMECORR' in header:
            self.timecorr[frameno] = header['TIMECORR']
        if 'CADENCEN' in header:
            self.cadenceno[frameno] = header['CADENCEN']
        if 'QUALITY' in header:
            self.quality[frameno] = header['QUALITY']
        if 'POS_CORR1' in header:
            self.pos_corr1[frameno] = header['POS_CORR1']
        elif 'PTGDIFFX' in header:
            self.pos_corr1[frameno] = header['PTGDIFFX']
        if 'POS_CORR2' in header:
            self.pos_corr2[frameno] = header['POS_CORR2']
        elif 'PTGDIFFY' in header:
            self.pos_corr2[frameno] = header['PTGDIFFY']


    def _check_data(self):
        """Check the data before writing to a TPF for any obvious errors."""
        if len(self.time) != len(np.unique(self.time)):
            warnings.warn('The factory-created TPF contains cadences with '
                          'identical TIME values.', LightkurveWarning)
        if ~np.all(self.time == np.sort(self.time)):
            warnings.warn('Cadences in the factory-created TPF do not appear '
                          'to be sorted in chronological order.', LightkurveWarning)
        if np.sum(self.flux==self.flux) == 0:
            warnings.warn('The factory-created TPF does not appear to contain '
                          'non-zero flux values.', LightkurveWarning)

    def get_tpf(self, hdu0_keywords=None, ext_info=None, **kwargs):
        """Returns a TargetPixelFile object."""
        if hdu0_keywords is None:
            hdu0_keywords = {}
        if ext_info is None:
            ext_info = {}
        self._check_data()

        mission = None
        if 'MISSION' in hdu0_keywords.keys():
            mission = hdu0_keywords['MISSION']

        if (mission=='Kepler') or (mission=='K2'):
            return KeplerTargetPixelFile(self._hdulist(hdu0_keywords=hdu0_keywords,
                                        ext_info=ext_info), **kwargs)
        elif mission=='TESS':
            return TessTargetPixelFile(self._hdulist(hdu0_keywords=hdu0_keywords,
                                        ext_info=ext_info), **kwargs)
        else:
            return TargetPixelFile(self._hdulist(hdu0_keywords=hdu0_keywords,
                                        ext_info=ext_info), **kwargs)

    def _hdulist(self, hdu0_keywords, ext_info):
        """Returns an astropy.io.fits.HDUList object."""
        return fits.HDUList([self._make_primary_hdu(hdu0_keywords=hdu0_keywords),
                             self._make_target_extension(ext_info=ext_info),
                             self._make_aperture_extension()])

    def _header_template(self, extension):
        """Returns a template `fits.Header` object for a given extension."""
        template_fn = os.path.join(PACKAGEDIR, "data",
                                   "tpf-ext{}-header.txt".format(extension))
        return fits.Header.fromtextfile(template_fn)

    def _make_primary_hdu(self, hdu0_keywords):
        """Returns the primary extension (#0)."""
        hdu = fits.PrimaryHDU()
        # Copy the default keywords from a template file from the MAST archive
        tmpl = self._header_template(0)
        for kw in tmpl:
            hdu.header[kw] = (tmpl[kw], tmpl.comments[kw])
        # Override the defaults where necessary
        hdu.header['ORIGIN'] = "Unofficial data product"
        hdu.header['DATE'] = datetime.datetime.now().strftime("%Y-%m-%d")
        hdu.header['TELESCOP'] = "Kepler"
        hdu.header['CREATOR'] = "lightkurve.KeplerTargetPixelFileFactory"
        hdu.header['OBJECT'] = self.targetid
        hdu.header['KEPLERID'] = self.targetid
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

    def _make_target_extension(self, ext_info):
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

class KeplerTargetPixelFileFactory(TargetPixelFileFactory):
    """Same as TargetPixelFileFactory."""
    # Backwards compatibility
    def __init__(self, n_cadences, n_rows, n_cols, targetid="unnamed-target",
                 keywords=None):
        msg = '`KeplerTargetPixelFileFactory` is deprecated, please use ' \
                '`TargetPixelFileFactory` instead.'
        warnings.warn(msg, LightkurveWarning)
        super(KeplerTargetPixelFileFactory, self).__init__(n_cadences, n_rows, n_cols, targetid=targetid,
                     keywords=keywords)
