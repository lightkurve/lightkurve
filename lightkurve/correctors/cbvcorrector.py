"""Defines Corrector classes that utilize Kepler/K2/TESS Cotrending Basis Vectors.
"""
import logging

from tqdm import tqdm

import oktopus
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from astropy.io import fits as pyfits
from bs4 import BeautifulSoup
import urllib.request
import astropy.units as u

from .. import MPLSTYLE

from ..lightcurve import LightCurve, KeplerLightCurve
from ..lightcurvefile import KeplerLightCurveFile
from .corrector import Corrector
from ..utils import channel_to_module_output, validate_method
from .designmatrix import DesignMatrix, DesignMatrixCollection
from .regressioncorrector import RegressionCorrector

log = logging.getLogger(__name__)

# For Kepler/K2/TESS max number of stored CBVs has always been 16
DEFAULT_NUMBER_CBVS = 16

__all__ = ['DEFAULT_NUMBER_CBVS', 'CotrendingBasisVectors', 'KeplerCotrendingBasisVectors',
        'TessCotrendingBasisVectors', 'get_kepler_cbvs', 'get_tess_cbvs', 'CBVCorrector']

#*******************************************************************************
class CotrendingBasisVectors:
    """
    Defines a CotrendingBasisVectors class. Superclass for KeplerCotrendingBasisVectors and TessCotrendingBasisVectors

    Stores Cotrending Basis Vectors for the Kepler/K2/TESS missions.

    Use search_cbvs to find the appropriate FITs files and KeplerCBVFile and TessCBVFile to retrieve the CBV FITS files.

    Each CotrendingBasisVectors objects contains only ONE set of CBVs. Instantiate multiple objects to store multiple set of
    CBVS, for example to save all three multi-scale bands.

    Attributes
    ----------
    mission         : [str] ('Kepler', 'K2', 'TESS')
    quarter         : only for Kepler mission
    campaign        : only for K2 mission
    sector          : only for TESS mission
    module,output   : only for Kepler/K2
    camera,ccd      : only for TESS
    cbv_type        : [str ('SingleScale', 'MultiScale', 'Spike')
    band            : [int] MultiScale band number (invalid for other CBV types)
    cbv_indices     : [int array] List of CBVs extracted for FITS file, {'ALL' => extract all}
    cbv_array       : [np.ndarray] The basis vectors
    cadenceno       : [int list] Cadence indices
    gap_indicators  : [bool list]
    cbvEXTNAME      : [str] The EXTNAME extension in the header

    """

    #***
    validMissionOptions = ('Kepler', 'K2', 'TESS')
    validCBVTypes = ('SingleScale', 'MultiScale', 'Spike')

    #***

    def __init__(self, mission, cbv_type='SingleScale', band=None, cbv_indices='ALL'):

        # Check if a valid mission was passed
        try:
            self.validMissionOptions.index(mission)
            self.mission = mission
        except:
            raise ValueError('Invalid mission')

        # Check if a valid cbv_type was passed
        try:
            self.validCBVTypes.index(cbv_type)
            self.cbv_type = cbv_type
        except:
            raise ValueError('Invalid cbv_type')

        self.band = band

        # For Kepler/K2/TESS it's always been 16 CBVs
        # TODO: automate this based on the CBVs in the FITS file, to future-proof potential changes
        if (isinstance(cbv_indices, str) and (cbv_indices == 'ALL')):
            cbv_indices=np.arange(1,DEFAULT_NUMBER_CBVS+1)
        self.cbv_indices = cbv_indices

    def _cbvs_to_matrix(self, cbv_indices='ALL'):
        """ Converts cbv_array (which is a list of np.ndarrays, one for each
        CBV, to a two-dimensional ndarray where the columns are the CBVs

        Parameters
        ----------
        cbv_indices : list
            List of CBV vectors to use. {'ALL' => Use all}

        """
        if (isinstance(cbv_indices, str) and (cbv_indices == 'ALL')):
            cbv_indices=np.arange(1,DEFAULT_NUMBER_CBVS+1)
        nCBVs = len(self.cbv_array)
        cbv_indices = cbv_indices[cbv_indices<=nCBVs]
    
        # Keep in mind that the CBVs are 1-based indexing
        # So, subract 1 from indices!
        return np.array(self.cbv_array[cbv_indices-1]).T
 
    def plot_cbvs(self, cbv_indices='ALL', ax=None):
        """Plot the requested CBVs

            Does not plot gapped cadences

        Parameters
        ----------
        cbv_indices  : list of ints
                        The list of cotrending basis vectors to plot. For example:
                            [1, 2] will fit the first two basis vectors. 'ALL' => plot all
        ax          : matplotlib.pyplot.Axes.AxesSubplot
                        Matplotlib axis object. If `None`, one will be generated.

        Returns
        -------
        ax      : matplotlib.pyplot.Axes.AxesSubplot
                    Matplotlib axis object
        """
        with plt.style.context(MPLSTYLE):
            if (isinstance(cbv_indices, str) and (cbv_indices == 'ALL')):
                cbv_indices=np.arange(1,len(self.cbv_array)+1)
            cbvChosenLogicalArray = np.in1d(np.arange(1, len(self.cbv_array)+1), np.asarray(cbv_indices))

            if ax is None:
                _, ax = plt.subplots(1)

            for idx, cbv in enumerate(self.cbv_array[cbvChosenLogicalArray, :][:, :]):
                cbvIndex = cbv_indices[idx]
                # Do not plot gaps
                cbv[self.gap_indicators] = np.nan
                ax.plot(self.cadenceno, cbv-idx/10., label='{}'.format(cbvIndex))

            ax.set_yticks([])
            ax.set_xlabel('Cadence Number')

            if self.mission == 'Kepler':
                ax.set_title('Kepler CBVs (Quarter.Module.Output : {}.{}.{})'
                             ''.format(self.quarter, self.module, self.output))
            elif self.mission == 'K2':
                ax.set_title('K2 CBVs (Campaign.Module.Output : {}.{}.{})'
                             ''.format( self.campaign, self.module, self.output))
            elif self.mission == 'TESS':
                if (self.cbv_type == 'MultiScale'):
                    ax.set_title('TESS CBVs (Sector.Camera.CCD : {}.{}.{}, CBVType.Band : {}.{})'
                             ''.format(self.sector, self.camera, self.ccd, self.cbv_type, self.band),
                             fontdict={'fontsize': 9})
                else:
                    ax.set_title('TESS CBVs (Sector.Camera.CCD : {}.{}.{}, CBVType : {})'
                             ''.format(self.sector, self.camera, self.ccd, self.cbv_type))

            ax.grid(':', alpha=0.3)
            ax.legend()
        return ax

    def align(self, lc, trim_lc=False):
        """ Aligns the CBVs with a light curve. The lightCurve object might not have the same cadences as the CBVs. This
        will trim the CBVs to be aligned with the light curve. 

        This method will preferentially use the cadence number (lc.cadenceno) to perform the synchronization,
        but will revert to using cadence time if cadenceno is not available in the light curve, which is more prone to errors

        It will report a warning and not synchronize if the light curve contains cadences not in the CBVs, unless trim_lc=True, in whcih
        case the light curve will also be trimmed.

        Parameters
        ----------
            lc : LightCurve object
                The reference light curve to align to
            trim_lc : [bool] If True then also trim the light curve, if needed

        Returns
            lc : LightCurve object
                If trim_lc = True then the return light curve is also trimmed,
                if needed
        -------
        """

        if not isinstance(lc, LightCurve):
            raise Exception('<lc> must be a LightCurve class')


        if hasattr(lc, 'cadenceno'):

            lc_trim_mask = np.in1d(lc.cadenceno, self.cadenceno)
            if (np.any(np.logical_not(lc_trim_mask))):
                if (trim_lc):
                    # trim the light curve
                    lc = lc[lc_trim_mask]
                else:
                    log.warning('There are cadences in the light curve that are not in the CBVs. NO SYNCHRONIZATION OCCURED')


            trim_mask = np.in1d(self.cadenceno, lc.cadenceno)
            self.cbv_array      = self.cbv_array[:,trim_mask]
            self.cadenceno      = self.cadenceno[trim_mask]
            self.gap_indicators = self.gap_indicators[trim_mask]

        else:
            log.warning('Synchronization with cadence time stamps is not yet implemented. NO SYNCHRONIZATION OCCURED')

        return lc

    @staticmethod
    def _extract_cbvs_from_hdu_data(cbv_data, cbv_indices):
        """ STATIC method: Extracts the CBVs from the HDU[extName].data CBV data set

        Will remove all-zero CBVs.

        For internal use only

        Parameters
        ----------
        cbv_data : HDU extension data
                    The CBV data from the HDU extension
        cbv_indices : int-like
                    List of CBV indices to extract

        Returns
        -------
        cbv_array   : float array
                    The extracted CBVs in a ndarray list, all-zero CBVs removed
        cbv_indices : int array
                    List of CBV indices to extract all-zero CBVs removed
        """
        cbv_array = []
        indices_to_remove = []
        for idx, i in enumerate(cbv_indices):
            try:
                cbv = cbv_data.field('VECTOR_{}'.format(i))
                if (np.all(cbv == 0.0)):
                    # all-zero CBV, remove form list
                    raise Exception()
                cbv_array.append(cbv)
            except:
                # For CBV vectors that do not exist remove from cbv_indices list
                indices_to_remove.append(idx)
        cbv_indices = np.delete(cbv_indices, indices_to_remove)
        cbv_array = np.asarray(cbv_array)

        return cbv_array, cbv_indices
        
#***
class KeplerCotrendingBasisVectors(CotrendingBasisVectors):
    """ Sub-class for Kepler/K2 cotrending basis vectors


        Parameters
        ----------

    """

    #***
    validMissionOptions = ('Kepler', 'K2')
    validCBVTypes = ('SingleScale')

    #***

    def __init__(self, mission, HDU, module, output, cbv_indices='ALL'):
        """ Kepler/K2 CBVs are all in the same FITS file for each quarter/campaign, so, when intantiating the CBV object
        we must specify which module and output we desire. Only Single-Scale CBVs are stored for Kepler.
        """

        super(KeplerCotrendingBasisVectors, self).__init__(mission, cbv_type='SingleScale', band=None, cbv_indices=cbv_indices)
        del mission, cbv_indices # Force use of object attributes

        if (self.mission == 'Kepler'):
            self.quarter = HDU['Primary'].header['QUARTER']
            self.campaign = None
        elif (self.mission == 'K2'):
            self.campaign = HDU['Primary'].header['CAMPAIGN']
            self.quarter = None

        self.module = module
        self.output = output

        extName = 'MODOUT_{0}_{1}'.format(module, output)
        cbv_data = HDU[extName].data

        self.cadenceno      = HDU[extName].data['CADENCENO']
        self.gap_indicators = HDU[extName].data['GAPFLAG']
        self.cbvEXTNAME     = HDU[extName].header['EXTNAME']

        # Pull out each individual CBV
        self.cbv_array, self.cbv_indices = self._extract_cbvs_from_hdu_data(cbv_data, self.cbv_indices)

#***
class TessCotrendingBasisVectors(CotrendingBasisVectors):

    #***
    validMissionOptions = ('TESS')
    validCBVTypes = ('SingleScale', 'MultiScale', 'Spike')

    #***

    def __init__(self, HDU, cbv_type, band, cbv_indices='ALL'):
        """ TESS CBVs are in seperate FITS files for each camera.CCD, so camera.CCD is already specified in the
        TessCBVFile object, here we need to specify which CBV type and band is desired.
        """

        mission = HDU['PRIMARY'].header['TELESCOP']
        assert mission == 'TESS', 'This does not appear to be a TESS FITS HDU'

        super(TessCotrendingBasisVectors, self).__init__(mission, cbv_type=cbv_type, band=band, cbv_indices=cbv_indices)
        del mission, cbv_type, band, cbv_indices # Force use of object attributes

        self.sector = HDU['PRIMARY'].header['SECTOR']
        # Curiosly, camera and CCD are not in the primary header!
        self.camera = HDU[1].header['CAMERA']
        self.ccd = HDU[1].header['CCD']

        # Get the requested cbv_type
        switcher = {
            'SingleScale': 'CBV.single-scale.{}.{}'.format(self.camera, self.ccd),
            'MultiScale': 'CBV.multiscale-band-{}.{}.{}'.format(self.band,
                self.camera, self.ccd),
            'Spike': 'CBV.spike.{}.{}'.format(self.camera, self.ccd),
            'unknown': 'error'
            }
        extName = switcher.get(self.cbv_type, switcher['unknown'])
        if (extName == 'error'):
            raise Exception('Invalide cbv_type')


        cbv_data = HDU[extName].data

        self.cadenceno      = HDU[extName].data['CADENCENO']
        self.gap_indicators = HDU[extName].data['GAP']
        self.cbvEXTNAME     = HDU[extName].header['EXTNAME']

        # Pull out each individual CBV
        self.cbv_array, self.cbv_indices = self._extract_cbvs_from_hdu_data(cbv_data, self.cbv_indices)

#*******************************************************************************
# Functions

def get_kepler_cbvs (mission=('Kepler', 'K2', 'TESS'), quarter=None, campaign=None,
        channel=None, module=None, output=None, cbv_indices='ALL'):
    """ Searches the `public data archive at MAST <https://archive.stsci.edu>`
    for Kepler or K2 cotrending basis vectors.

    This function fetches the Cotrending Basis Vectors FITS HDU for the desired
    mission, quarter/campaign and channel or module/output, etc...
    and then extracts the requested basis vectors

    For Kepler/K2, the FITS files contain all channels in a single file per
    quarter/campaing.

    For Kepler extracts the DR25 CBVs

    Parameters
    ----------
    mission     : str, list of str
                    'Kepler' or 'K2'
    quarter or campaign : int, list of ints
                    Kepler Quarter or K2 Campaign.
    channel or module and output : int
                    Kepler/K2  requested channel or module and output
                    Must provide either channel, or module and outout, 
                    but not both
    cbv_indices : int array
                    List of CBVs extracted from FITS file, 1-Based {'ALL' => extract all}

    Returns
    -------
    result : :class:`KeplerCotrendingBasisVectors` object
        Object detailing the data products found.

    Examples
    --------
    This example will read in the CBVs for Kepler quarter 8,
    and then extract the first 8 CBVs for module.output 16.4

        >>> cbvs = get_kepler_cbvs(mission='Kepler', quarter=8, module=16, output=4,   # doctest: +SKIP
        >>>     cbv_indices=np.arange(1,9))                                     # doctest: +SKIP

    """

    #***
    # Validate inputs
    # Make sure only the appropriate arguments are passed
    if (mission == 'Kepler'):
        assert  isinstance(quarter, int), 'quarter must be passed for Kepler mission'
        assert  campaign is None, 'campaign must not be passed for Kepler mission'
    elif (mission == 'K2'):
        assert  isinstance(campaign, int), 'campaign must be passed for K2 mission'
        assert  quarter is None,  'quarter must not be passed for K2 mission'
    else:
        raise Exception('Unknown mission type')

    # CBV FITS files use module/output, not channel
    # So if channel is passed, convert to module/output
    if (isinstance(channel, int)):
        assert  module is None, 'module must NOT be passed if channel is passed'
        assert  output is None, 'output must NOT be passed if channel is passed'
        module, output = channel_to_module_output(channel)
        channel = None
    else:
        assert  module is not None, 'module must be passed'
        assert  output is not None, 'output must be passed'

    if (mission == 'Kepler'):
        cbvBaseUrl = "http://archive.stsci.edu/missions/kepler/cbv/"
    elif (mission == 'K2'):
        cbvBaseUrl = "http://archive.stsci.edu/missions/k2/cbv/"

    try:     
        soup = BeautifulSoup(requests.get(cbvBaseUrl).text, 'html.parser')
        cbv_files = [fn['href'] for fn in soup.find_all('a') if fn['href'].endswith('fits')]
 
        if mission == 'Kepler':
            quarter = 'q{:02}'.format(quarter)
            for cbv_file in cbv_files:
                if quarter + '-d25' in cbv_file:
                    break
        elif mission == 'K2':
            campaign = 'c{:02}'.format(campaign)
            for cbv_file in cbv_files:
                if campaign in cbv_file:
                    break

        kepler_cbv_url = cbvBaseUrl + cbv_file
        hdu = pyfits.open(kepler_cbv_url)

        cbvs = KeplerCotrendingBasisVectors(mission, hdu, module, output, 
                cbv_indices=cbv_indices)

        return cbvs

    except:
        raise Exception('CBVS were not found')

def get_tess_cbvs (sector=None, camera=None,
        CCD=None, cbv_type='SingleScale', band=None, cbv_indices='ALL'):

    """ Searches the `public data archive at MAST <https://archive.stsci.edu>`
    for TESS cotrending basis vectors.

    This function fetches the Cotrending Basis Vectors FITS HDU for the desired
    cotrending basis vectors.

    For TESS, each CCD CBVs are stored in a seperate FITS files.

    Parameters
    ----------
    sector : int, list of ints
                    TESS Sector number.
    camera and CCD : int, list of ints
                    TESS camera and CCD
    cbv_type    : str
                    'SingleScale' or 'MultiScale' or 'Spike'
    band        : int
                    Multi-scale band number
    cbv_indices : int array
                    List of CBVs extracted from FITS file, 1-Based {'ALL' => extract all}

    Returns
    -------
    result : :class:`TessCotrendingBasisVectors` object
        Object detailing the data products found.

    Examples
    --------
    This example will read in the CBVs for TESS Sector 10 Camera.CCD 2.4
    and then extract the first 6 CBVs of multi-scale band 2

        >>> cbvs = get_tess_cbvs(sector=10, camera=2, CCD=4, # doctest: +SKIP
        >>>     cbv_type='MultiScale', band=2, cbv_indices=np.arange(1,7)) # doctest: +SKIP
    """

    # The easiest way to obtain a link to the CBV file for a TESS Sector and camera.CCD is
    # 
    # 1. Download the bulk download curl script (with a predictable url) for the desired sector and search it for the camera.CCD I need
    # 2. Download the CBV FITS file based on the link in the curl script
    #
    # The bulk download curl links have urls such as:
    #
    # https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_17_cbv.sh
    #
    # Then the individual CBV files foudn in the curl file have urls such as:
    #
    # https://archive.stsci.edu/missions/tess/ffi/s0017/2019/279/1-1/tess2019279210107-s0017-1-1-0161-s_cbv.fits

    #***
    # Validate inputs
    # Make sure only the appropriate arguments are passed
    assert  isinstance(sector, int),    'sector must be passed for TESS mission'
    assert  isinstance(camera, int),    'camera must be passed'
    assert  isinstance(CCD, int),       'CCD must be passed'
    if cbv_type == 'MultiScale':
        assert  isinstance(band, int),  'band must be passed for multi-scale CBVs'
    else:
        assert  band is None,  'band must NOT be passed for single-scale or spike CBVs'
        
    curlBaseUrl = 'https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_'
    curlEndUrl = '_cbv.sh'
    curlUrl = curlBaseUrl + str(sector) + curlEndUrl

    # This is the string to search for in the curl script file
    # Pad the sector number with a first '0' if less than 10
    # TODO: figure out a way to pad an interger number with forward zeros
    # without needing a conditional
    sector = int(sector)
    if (sector < 10):
        curlSearchString = 's000' + str(sector) + '-' + str(camera) + '-' + str(CCD) + '-'
    elif (sector > 99):
        # TESS will be blessed if it gets to more than 99 sectors!
        raise Exception('Only up to 99 Sectors is currently supported')
    else:
        curlSearchString = 's00' + str(sector) + '-' + str(camera) + '-' + str(CCD) + '-'

    try: 

        # 1. Read in the relevent curl script file and find the line for the CBV data we are looking for
        data = urllib.request.urlopen(curlUrl)
        foundIndex = None
        for line in data:
            strLine = str(line)
            try:
                foundIndex = strLine.index(curlSearchString) # str.index will error when not found
                break
            except Exception:
                pass # continue searching
        if (foundIndex is None):
            raise Exception('CBV FITS file not found')

        # extract url from strLine
        htmlStartIndex = strLine.find('https:')
        htmlEndIndex = strLine.rfind('fits')
        tess_cbv_url  = strLine[htmlStartIndex:htmlEndIndex+4] # Add 4 for length of 'fits' string
        
        hdu = pyfits.open(tess_cbv_url)
            
        # Check that this is a TESS CBV FITS file
        mission = hdu['Primary'].header['TELESCOP']
        validate_method(mission, ['tess'])

        cbvs = TessCotrendingBasisVectors(hdu, cbv_type, band, cbv_indices=cbv_indices)

        return cbvs
            

    except:
        raise Exception('CBVS were not found')


#*******************************************************************************
# Correctors


class CBVCorrector(RegressionCorrector):
    """ Class for removing systematics using CBV correctors for Kepler/K2/TESS



    Attributes
    ----------
    lc  : LightCurve
        The light curve to correct
    cbvs    : CetrendingBasisVectors list
        The retrieved CBVs, can contain multiple types
    cbv_type : str list
        List of CBV types to use in correction {'ALL' => Use all}
    cbv_indices : list of lists
        List of CBV vectors to use in each of cbv_type passed. {'ALL' => Use all}
    cbv_design_matrix : DesignMatrix
        The retrieved CBVs ported into a DesignMatrix object
    extra_design_matrix : DesignMatrix
        An extra design matrix to include in the fit with the CBVs
    design_matrix_collection   : DesignMatrixCollection
        The design matrix collection composed of cbv_design_matrix and extra_design_matrix
    corrected_lc : LightCurve
        The returned light curve from regression_corrector.correct


    """

    def __init__(self, lc):
        """ Constructor for CBVClass objects

        This constructor will retrieve all relevant CBVs from MAST and then
        align them with the passed in light curve.

        For TESS we have the option to load multiple CBV types.
        
        Parameters
        ----------
        lc  : LightCurve
            The light curve to correct

        Examples
        --------
        """

        if not isinstance(lc, LightCurve):
            raise Exception('<lc> must be a LightCurve class')

        # Call the RegresssionCorrector Constructor
        super(CBVCorrector, self).__init__(lc)

        #***
        # Retrieve all relevant CBVs from MAST
        # TODO: create CBV collection class
        cbvs = []

        if self.lc.mission == 'Kepler':
            cbvs.append(get_kepler_cbvs(mission=self.lc.mission, quarter=self.lc.quarter,
                    channel=self.lc.channel, cbv_indices='ALL'))
            self.cbv_indices = cbvs.cbv_indices
        elif self.lc.mission == 'K2':
            cbvs.append(get_kepler_cbvs(mission=self.lc.mission, campaign=self.lc.campaign,
                    channel=self.lc.channel, cbv_indices='ALL'))
            self.cbv_indices = cbvs.cbv_indices
        elif self.lc.mission == 'TESS':
            # For TESS we load multiple CBV types

            cbvs.append(get_tess_cbvs(sector=self.lc.sector,
                camera=self.lc.camera, CCD=self.lc.ccd, cbv_type='SingleScale',
                cbv_indices='ALL'))

            # TODO: loop over multi-scale CBV groups to searhc all in CNBV FITS
            # file
            cbvs.append(get_tess_cbvs(sector=self.lc.sector,
                camera=self.lc.camera, CCD=self.lc.ccd, cbv_type='MultiScale',
                band=1, cbv_indices='ALL'))

            cbvs.append(get_tess_cbvs(sector=self.lc.sector,
                camera=self.lc.camera, CCD=self.lc.ccd, cbv_type='MultiScale',
                band=2, cbv_indices='ALL'))

            cbvs.append(get_tess_cbvs(sector=self.lc.sector,
                camera=self.lc.camera, CCD=self.lc.ccd, cbv_type='MultiScale',
                band=3, cbv_indices='ALL'))

            cbvs.append(get_tess_cbvs(sector=self.lc.sector,
                camera=self.lc.camera, CCD=self.lc.ccd, cbv_type='Spike',
                cbv_indices='ALL'))

        else:
            raise ValueError('Unknown mission type')

        for idx in np.arange(len(cbvs)):
            if (not isinstance(cbvs[idx], CotrendingBasisVectors)):
                raise Exception('CBVs could not be loaded. CBVCorrector must exit')

        # Align the CBVs with the lightcurve flux using the cadence numbers
        for idx in np.arange(len(cbvs)):
            self.lc = cbvs[idx].align(self.lc, trim_lc=True)

        self.cbvs = cbvs


    def correct(self, cbv_type='SingleScale', cbv_indices='ALL', alpha=1e-20, l1_ratio=0.01, ext_dm=None, cadence_mask=None):
        """ Performs the correction using RegressionCorrector methods

        This method will assemble the full design matrix collection composed of
        cbv_design_matrix and extra_design_matrix. Then use the super-class
        RegressionCorrector.correct to perform the correction

        TODO: A whole bunch! This is a shell of a method. Just to demonstrate
        basic functionality. We should consider allowing for standard
        regaulrization methods such as Ridge Regression (L2 Norm) and Lasso (L1
        Norm).  Maybe even BIC or AIC methods also.

        Eventually, the plan is to use over-fitting and under-fitting goodness
        metrics to constrain the regression fit.

        Parameters
        ----------
        cbv_type : str list
            List of CBV types to use
        cbv_indices : list of lists
            List of CBV vectors to use in each passed cbv_type. {'ALL' => Use all}
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to also be used in the fit
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.

        Examples
        --------
            >>> cbv_type = ['SingleScale', 'Spike']
            >>> cbv_indices = [np.arange(1,9), 'ALL']
            >>> corrected_lc = cbvCorrector.correct(cbv_type=cbv_type,
                    cbv_indices=cbv_indices,  )
        """
        
        # Perform all the preparatory stuff common to all correct methods
        self._correct_initialization(cbv_type=cbv_type,
                cbv_indices=cbv_indices, ext_dm=ext_dm, cadence_mask=cadence_mask)

            
        # Use RegressionCorrector.correct for the actual fitting
       #super(CBVCorrector, self).correct(design_matrix)

        from sklearn import linear_model

        # Use Scikit-learn ElasticNet
        self.regressor = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                normalize=True)

        X = self.design_matrix_collection.values
        y = self.lc.flux

        metric = self._apply_fit(X, y)

        return self.corrected_lc

    def correct_optimizer(self, cbv_type='SingleScale', cbv_indices='ALL', alpha_init=1e-20, l1_ratio=0.01, ext_dm=None,
            cadence_mask=None, max_iter=100, target_score=0.0):
        """ Performs the correction using RegressionCorrector methods

        This method will adjust the regularization penalty term based on the introduced
        noise metric. l1_ratio is not optimized.

        It uses a basic gradient decent method hit the target introduced noise
        metric.

        Parameters
        ----------
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to also be used in the fit
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        alphas_init : float
            The initial regularization penalty term. Start with a small number.
        l1_ratio : float
            The ElasticNet mixing parameter
        target_score : float
            Target Over-fitting metric score


        """

        # Perform all the preparatory stuff common to all correct methods
        self._correct_initialization(cbv_type=cbv_type,
                cbv_indices=cbv_indices, ext_dm=ext_dm, cadence_mask=cadence_mask)


        from sklearn import linear_model

        X = self.design_matrix_collection.values
        y = self.lc.flux
            
        # Use Scikit-learn ElasticNet
        self.regressor = linear_model.ElasticNet(alpha=alpha_init,
                l1_ratio=l1_ratio, normalize=True)

        alpha = alpha_init
        alpha_step_factor = 1.1 # being by increasing alpha by 10%
        for idx in np.arange(max_iter):

            metric = self._apply_fit(X, y, alpha)

            # Check goodness metric
            if (metric > target_score):
                # Over-fitting occured, increase alpha
                alpha *= alpha_step_factor
            else:
                # We're good, exit loop
                break
        
        self.introduced_noise_score = metric


        return self.corrected_lc




    def over_fitting_metric(self):
        """ Uses a LombScarglePeriodogram to assess the change in broad-band
        power in a corrected light curve to measure degree of over-fitting

        This function expects a median normalized light curve
        """

        # TODO: handle gaps and/or masks

        pgOrig = self.lc.to_periodogram(freq_unit=u.microHertz, maximum_frequency=400, minimum_frequency=10)
        pgCorrected = self.corrected_lc.to_periodogram(freq_unit=u.microHertz, maximum_frequency=400, minimum_frequency=10)

        # Get an esitmate of the PSD at the uncertainties limit
        # The raw and corrected uncertainties should be essentially identical so use the corrected
        # TODO: the periodogram of WGN should be analytical to compute!
        nNonGappedCadences = len(self.lc.flux)
        meanCorrectedUncertainties = np.mean(self.corrected_lc.flux_err)
        WGNCorrectedUncert = (np.random.randn(nNonGappedCadences,1) * meanCorrectedUncertainties).T[0]
        model_err   = np.zeros(len(self.lc.flux))
        noise_lc = LightCurve(self.lc.time, WGNCorrectedUncert, model_err)
        pgCorrectedUncert = noise_lc.to_periodogram(freq_unit=u.microHertz, maximum_frequency=400, minimum_frequency=10)
        meanCorrectedUncertPower = np.mean(np.array(pgCorrectedUncert.power))
        
        # Compute the change in power
        pgChange = np.array(pgCorrected.power) - np.array(pgOrig.power)

        # If no increase in power in ANY bands then return a perfect loss
        # function
        if (len(np.nonzero(pgChange>0.0)[0]) == 0):
            return 0.0
        
        # We are only concerned with bands where the power increased so
        # when(pgCorrected - pgRaw) > 0 
        # Normalize by the noise in the uncertainty
        # We want the goodness to begin to degrade when the introduced noise is greater than the uncertainties.
        # So, when DeltaNoise > 0.5 (given twiceSigmoidInv defn.)
        metric = np.sum(pgChange[pgChange>0.0]) / ((len(np.nonzero(pgChange>0.0)[0]))*meanCorrectedUncertPower)

        
      # #****************
      # # This converts the loss function into a goodness metric
      # # Adjust the decay rate of the sigmoid so that it's not too steep of a decent.
      # # A noiseScale of about 0.05 looks good.
      # noiseScale = 0.05
      # metric *= noiseScale
      # if (metric < 0.0):
      #     metric = 0.0

      # # We want the goodness to span (0,1] so take the inverse of each component using a sigmoid
      # # Use twice an inverse sigmoid to get a [0,1] range from a [0,inf) range
      # def sigmoidInv(x): return 2.0 / (1 + np.exp(x))
      # metric     = sigmoidInv(metric)

        return metric

    def _correct_initialization(self, cbv_type='SingleScale', cbv_indices='ALL',
            ext_dm=None, cadence_mask=None):
        """ Performs all the preperatory needs before applying a correct method.

        This helper function is used so that multiple correct methods can be used
        without the need to repeat preparatory code.

        Does things like sets up the design matrix

        """

        if (self.lc.mission in ['Kepler', 'K2']):
            assert cbv_type is not 'SingleScale', 'cbv_type must be Single-Scale for Kepler and K2 missions'

        if (isinstance(cbv_type, list)):
            if (not self.lc.mission == 'TESS'):
                raise Exception('Multiple CBV types are only allowed for TESS')
            if (not len(cbv_type) == len(cbv_indices)):
                raise Exception('cbv_type and cbv_indices must be the same list length')


        if cadence_mask is None:
            self.cadence_mask = np.ones(len(self.lc.time), bool)
        else:
            self.cadence_mask = np.copy(cadence_mask)

        #***
        # Create the design matrix collection with CBVs, plus extra passed basis vectors

        # If any DesignMatrix was passed then store it
        self.extra_design_matrix = ext_dm

        # Create a CBV design matrix for each CBV sets requested
        self.cbv_design_matrix = []

        # Loop through all the stored CBVs and find the ones matching the
        # requested cbv_type list
        for idx in np.arange(len(cbv_type)): 
            for cbvs in self.cbvs:

                # Temporarily copy the cbv_indices requested
                cbv_idx_loop = cbv_indices[idx]

                # If requesting 'ALL' CBVs then set to max default number
                # Remember, cbv indices is 1-based!
                if (isinstance(cbv_idx_loop, str) and (cbv_idx_loop == 'ALL')):
                    cbv_idx_loop=np.arange(1,DEFAULT_NUMBER_CBVS+1)
                # Trim to nCBVs in cbvs
                nCBVs = len(cbvs.cbv_array)
                cbv_idx_loop = cbv_idx_loop[cbv_idx_loop<=nCBVs]

                if cbv_type[idx].find('MultiScale') >= 0:
                    # Find the correct band if this is a multi-scale CBV set
                    band = cbv_type[idx][-1]
                    if (cbvs.cbv_type in cbv_type[idx] and cbvs.band == band):
                        cbv_index_names = [cbv_index for cbv_index in
                                cbv_idx_loop]
                        self.cbv_design_matrix.append(DesignMatrix(cbvs._cbvs_to_matrix(cbv_idx_loop),
                            name=cbv_type[idx], columns=cbv_index_names))
                else:
                    if (cbvs.cbv_type in cbv_type[idx]):
                        cbv_index_names = [cbv_index for cbv_index in
                                cbv_idx_loop]
                        self.cbv_design_matrix.append(DesignMatrix(cbvs._cbvs_to_matrix(cbv_idx_loop),
                            name=cbvs.cbv_type, columns=cbv_index_names))

        # Create the full design matrix collection
        if self.extra_design_matrix is not None:
            dm_to_flatten = [[cbv_dm for cbv_dm in self.cbv_design_matrix], [self.extra_design_matrix]]
            flattened_dm_list = [item for sublist in dm_to_flatten for item in sublist]
        else:
            dm_to_flatten = [[cbv_dm for cbv_dm in self.cbv_design_matrix]]
            flattened_dm_list = [item for sublist in dm_to_flatten for item in sublist]
        self.design_matrix_collection = DesignMatrixCollection(flattened_dm_list)
        self.design_matrix_collection._validate()


    def _apply_fit(self, X, y, alpha=None):
        """ Helper function to apply the regressor fit and set all values in
        object.

        Parameters
        ----------
            X
            y
            alpha   : float
                If passed then set the alpha penalty term to this value

        Returns
        -------
        metric  : float
            Over-fitting metric
        """
        
        if alpha is not None:
            self.regressor.alpha = alpha

        self.regressor.fit(X, y)

        model_flux  = np.dot(X, self.regressor.coef_)
        model_err   = np.zeros(len(model_flux))
        
        self.coefficients = self.regressor.coef_
        
        self.model_lc = LightCurve(self.lc.time, model_flux, model_err)
        self.corrected_lc = self.lc.copy()
        self.corrected_lc.flux = self.lc.flux - self.model_lc.flux
        self.corrected_lc.flux_err = (self.lc.flux_err**2 + model_err**2)**0.5
        self.diagnostic_lightcurves = self._create_diagnostic_lightcurves()
            
        metric = self.over_fitting_metric()

        return metric

            

#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
# CANDIDATE FOR REMOVAL
# Do we wish to retain any of this functionality? Or is it superseeded by
# RegressionCorrector?
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************

class OldKeplerCBVCorrector(Corrector):
    """Remove systematic trends from Kepler light curves by fitting
    Cotrending Basis Vectors (CBVs).

    .. math::

        \arg \min_{\bm{\theta} \in \Theta} \sum_{t}|f_{SAP}(t) - \sum_{j=1}^{n}\theta_j v_{j}(t)|^p, p>0, p \in \mathbb{R}

    Attributes
    ----------
    lc : KeplerLightCurveFile, KeplerLightCurve object or str
        An instance from KeplerLightCurveFile or a path for the .fits
        file of a NASA's Kepler/K2 light curve.
    likelihood : oktopus.Likelihood subclass
        A class that describes a cost function.
        The default is :class:`oktopus.LaplacianLikelihood`, which is tantamount
        to the L1 norm.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from lightkurve import KeplerCBVCorrector, KeplerLightCurveFile
    >>> fn = ("https://archive.stsci.edu/missions/kepler/lightcurves/"
    ...       "0084/008462852/kplr008462852-2011073133259_llc.fits") # doctest: +SKIP
    >>> cbv = KeplerCBVCorrector(fn) # doctest: +SKIP
    Downloading https://archive.stsci.edu/missions/kepler/lightcurves/0084/008462852/kplr008462852-2011073133259_llc.fits [Done]
    >>> cbv_lc = cbv.correct() # doctest: +SKIP
    Downloading http://archive.stsci.edu/missions/kepler/cbv/kplr2011073133259-q08-d25_lcbv.fits [Done]
    >>> sap_lc = KeplerLightCurveFile(fn).SAP_FLUX # doctest: +SKIP
    >>> plt.plot(sap_lc.time, sap_lc.flux, 'x', markersize=1, label='SAP_FLUX') # doctest: +SKIP
    >>> plt.plot(cbv_lc.time, cbv_lc.flux, 'o', markersize=1, label='CBV_FLUX') # doctest: +SKIP
    >>> plt.legend() # doctest: +SKIP
    """

    def __init__(self, lc, likelihood=oktopus.LaplacianLikelihood, prior=oktopus.LaplacianPrior):
        self.lc = lc
        if not hasattr(self.lc, 'channel'):
            raise ValueError('Input must have a `channel` attribute.')
        self.likelihood = likelihood
        self.prior = prior
        self._ncbvs = DEFAULT_NUMBER_CBVS  # default number of cbvs for Kepler/K2

        # Get the CBVs from MAST
        if self.lc.mission == 'Kepler':
            cbvs = get_kepler_cbvs(mission=self.lc.mission, quarter=self.lc.quarter,
                    channel=self.lc.channel, cbv_indices='ALL')
        elif self.lc.mission == 'K2':
            cbvs = get_kepler_cbvs(mission=self.lc.mission, campaign=self.lc.campaign,
                    channel=self.lc.channel, cbv_indices='ALL')

        self.cbvs = cbvs


        # Align the CBVs with the lightcurve flux using the cadence numbers
        cbvs.align(self.lc, trim_lc=True)


    @property
    def lc(self):
        return self._lc

    @lc.setter
    def lc(self, value):
        # this enables `lc` to be either a string
        # or an object from KeplerLightCurveFile
        if isinstance(value, str):
            self._lc = KeplerLightCurveFile(value).PDCSAP_FLUX
        elif isinstance(value, KeplerLightCurveFile):
            self._lc = value.SAP_FLUX
        elif isinstance(value, KeplerLightCurve):
            self._lc = value
        else:
            raise ValueError("lc must be either a string, a KeplerLightCurve or a"
                             " KeplerLightCurveFile instance, got {}.".format(value))

    @property
    def coeffs(self):
        """
        Returns the fitted coefficients.
        """
        return self._coeffs

    @property
    def opt_result(self):
        """
        Returns the result of the optimization process.
        """
        return self._opt_result

    def correct(self, cbv_indices=(1, 2), method='powell', options=None):
        """
        Correct the SAP_FLUX by fitting a number of cotrending basis vectors
        `CBVs`.

        Parameters
        ----------
        cbv_indices : list of ints
            The list of cotrending basis vectors to fit to the data. For example,
            [1, 2] will fit the first two basis vectors.
        method : str
            Numerical optimization method. See scipy.optimize.minimize for the
            full list of methods.
        options : dict
            Dictionary of options to be passed to scipy.optimize.minimize.
        """
        if options is None:
            options = {}
        median_flux = np.nanmedian(self.lc.flux)
        norm_flux = self.lc.flux / median_flux - 1
        norm_err_flux = self.lc.flux_err / median_flux

        # Trim down to the correct number of cbvs
        clip = np.in1d(np.arange(1, len(self.cbvs.cbv_array)+1), np.asarray(cbv_indices))
        def mean_model(*theta):
            coeffs = np.asarray(theta)
            return np.dot(coeffs, self.cbvs.cbv_array[clip, :][:, :])

        prior = self.prior(mean=np.zeros(len(cbv_indices)), var=16.)
        likelihood = self.likelihood(data=norm_flux, mean=mean_model,
                                     var=norm_err_flux)
        x0 = likelihood.fit(x0=prior.mean, method=method, options=options).x
        posterior = oktopus.Posterior(likelihood=likelihood, prior=prior)

        self._opt_result = posterior.fit(x0=x0, method=method,
                                         options=options)
        self._coeffs = self._opt_result.x
        flux_hat = self.lc.flux - median_flux * mean_model(self._coeffs)
        clc = self.lc.copy()
        clc.flux = flux_hat.reshape(-1)
        return clc

    def get_cbvs_list(self):
        """Returns the subsequence of subsequent CBVs that maximizes
        Bayes' factor [1]_.

        Returns
        -------
        cbv_list : list
            Subsequence of subsequent CBVs that maximizes the Bayes' factor.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Bayes_factor
        """

        self.bayes_factor, cost = [], []  # bayes_factor here is actually the
                                          # negative log of the bayes factor
        self.correct(cbvs=[1], options={'xtol': 1e-6, 'ftol': 1e-6, 'maxfev': 2000})
        cost.append(self.opt_result.fun)
        for n in tqdm(range(2, self._ncbvs+1)):
            cbv_list = list(range(1, n+1))
            self.correct(cbv_list, options={'xtol': 1e-6, 'ftol': 1e-6, 'maxfev': 2000})
            cost.append(self.opt_result.fun)
            # cost is the negative log of the posterior evaluated at the
            # Maximum A Posterior Probability (MAP) estimator
            self.bayes_factor.append((cost[n-2] - cost[n-1]))
            # so cost[n-2] - cost[n-1] = -log(p1) + log(p2) = log(p2/p1)
            # where p1 is the posterior probability (evaluated at the MAP)
            # for the model with n-2 cbvs and p2 is the posterior probability
            # also evaluated at the MAP for the model with n-1 cbvs
        k = np.argmin(self.bayes_factor)
        # transform to get the actual Bayes factor
        self.bayes_factor = np.exp(-np.array(self.bayes_factor))
        # the k+2 here comes from the fact that Python indexes begin
        # from 0 and we count CBVs starting from 1 and also
        # note that range(1, k) equals the interval [1, k), which excludes k.
        return list(range(1, k+2))

