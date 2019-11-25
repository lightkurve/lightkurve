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

        from .targetpixelfile import TargetPixelFile, KeplerTargetPixelFile, TessTargetPixelFile

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

    @staticmethod
    def _header_template(extension):
        """Returns a template `fits.Header` object for a given extension."""
        template_fn = os.path.join(PACKAGEDIR, "data",
                                   "tpf-ext{}-header.txt".format(extension))
        return fits.Header.fromtextfile(template_fn)

    def _make_primary_hdu(self, hdu0_keywords):
        """Returns the primary extension (#0)."""
        hdu = fits.PrimaryHDU()
        # Copy the default keywords from a template file from the MAST archive
        tmpl = TargetPixelFileFactory._header_template(0)
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


    @staticmethod
    def from_fits_images(images, position, size=(11, 11), extension=1,
                         targetid="unnamed-target", hdu0_keywords=None, **kwargs):
        """Creates a new Target Pixel File from a set of images.

        This method is intended to make it easy to cut out targets from full
        frame images including but not limited to those from, Kepler/K2
        "superstamps", TESS FFIs, and Spitzer point-and-stare observations.
        This method may work on point-and-step or "dithered" observation patterns,
        but has not been tested, and may fail if the target leaves the frame
        entirely.

        Parameters
        ----------
        images : list of str, or list of fits.ImageHDU objects
            Sorted list of FITS filename paths or ImageHDU objects to get
            the data from.
        position : astropy.SkyCoord
            Position around which to cut out pixels.
        size : (int, int)
            Dimensions in pixels (cols, rows) to cut out around `position`.
        extension : int or str
            If `images` is a list of filenames, provide the extension number
            or name to use. Default: 0.
        targetid : int or str
            Unique identifier of the target to be recorded in the TPF.
        hdu0_keywords : dict
            Additional keywords to add to the first header file.
        **kwargs : dict
            Extra arguments to be passed to the `TargetPixelFile` constructor.

        Returns
        -------
        tpf : TargetPixelFile
            A new Target Pixel File assembled from the images.
        """
        if len(images) == 0:
            raise ValueError('One or more images must be passed.')
        if not isinstance(position, SkyCoord):
            raise ValueError('Position must be an astropy.coordinates.SkyCoord.')
        if hdu0_keywords is None:
            hdu0_keywords = {}

        header_keywords = ['MISSION', 'TELESCOP', 'INSTRUME', 'QUARTER',
                          'CAMPAIGN', 'CHANNEL', 'MODULE', 'OUTPUT', 'BMJD_OBS',
                          'EXPTIME', 'BUNIT', 'RA_REF', 'DEC_REF']
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
        if size is None:
            size = _open_image(images[0], extension).data.shape

        # Find middle image to use as a WCS reference
        try:
            mid_hdu = _open_image(images[int(len(images) / 2) - 1], extension)
            wcs_ref = WCS(mid_hdu, relax=True)
            column_ref, row_ref = wcs_ref.all_world2pix(
                np.asarray([[position.ra.deg], [position.dec.deg]]).T, 0)[0]
        except Exception as e:
            raise e

        # Get some basic keywords
        for kw in header_keywords:
            if kw in mid_hdu.header:
                if not isinstance(mid_hdu.header[kw], Undefined):
                    carry_keywords[kw] = mid_hdu.header[kw]
        if ('MISSION' not in carry_keywords) and ('TELESCOP' in carry_keywords):
            carry_keywords['MISSION'] = carry_keywords['TELESCOP']

        allkeys = hdu0_keywords.copy()
        allkeys.update(carry_keywords)

        # Create a factory and set default keyword values based on the middle image
        factory = TargetPixelFileFactory(n_cadences=len(images),
                                         n_rows=size[0],
                                         n_cols=size[1],
                                         targetid=targetid)

        factory.keywords = mid_hdu.header

        for idx, img in tqdm(enumerate(images), total=len(images)):
            hdu = _open_image(img, extension)

            # Get positional shift of the image compared to the reference WCS
            wcs_current = WCS(hdu.header, relax=True)
            column_current, row_current = wcs_current.all_world2pix(
                np.asarray([[position.ra.deg], [position.dec.deg]]).T, 0)[0]

            with warnings.catch_warnings():
                # Using `POS_CORR1` as a header keyword violates the FITS
                # standard for being too long, but we use it for consistency
                # with the TPF column name.  Hence we ignore the warning.
                warnings.simplefilter("ignore", AstropyWarning)
                hdu.header['POS_CORR1'] = column_current - column_ref
                hdu.header['POS_CORR2'] = row_current - row_ref

            if position is None:
                cutout = hdu
            else:
                cutout = Cutout2D(hdu.data, position, wcs=wcs_ref,
                                  size=size, mode='partial')

            factory.add_cadence(frameno=idx, flux=cutout.data, header=hdu.header)

        # Add custom TPF header information needed for WCS purposes
        ext_info = {}
        ext_info.update(cutout.wcs.to_header())
        TFORM_suffix = {4:'J', 5:'E', 6:'E', 7:'E', 8:'E', 9:'E'}

        # Compute the distance from the star to the TPF lower left corner
        # That is approximately half the TPF size, with an adjustment factor if the star's pixel
        #    position gets rounded up or not.
        # The first int is there so that even sizes always round to one less than half of their value

        half_tpfsize_col = int((size[0] - 1) / 2.) + (int(round(column_ref)) - int(column_ref)) * ((size[0] + 1) % 2)
        half_tpfsize_row = int((size[1] - 1) / 2.) + (int(round(row_ref)) - int(row_ref)) * ((size[1] + 1) % 2)

        refpixels = {val:factory.keywords['CRVAL{}P'.format(val)] \
                    if 'CRVAL{}P'.format(val) in factory.keywords.keys() \
                    else 0 for val in [1,2]}  #TODO spot check the zero default

        # TPF contains multiple data columns that require WCS
        for m in [4, 5, 6, 7, 8, 9]:
            ext_info["TFORM{}".format(m)] = '{}{}'.format(size[0] * size[1], TFORM_suffix[m])
            ext_info['TDIM{}'.format(m)] = '({},{})'.format(size[0], size[1])
            ext_info['1CRV{}P'.format(m)] = int(round(column_ref)) - half_tpfsize_col + refpixels[1] - 1
            ext_info['2CRV{}P'.format(m)] = int(round(row_ref)) - half_tpfsize_row + refpixels[2] - 1

        return factory.get_tpf(hdu0_keywords=allkeys, ext_info=ext_info, **kwargs)


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
