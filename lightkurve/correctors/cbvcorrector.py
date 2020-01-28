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

__all__ = ['CotrendingBasisVectors', 'KeplerCotrendingBasisVectors',
        'TessCotrendingBasisVectors', 'get_cbvs', 'CBVCorrector']

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

    def _cbvs_to_matrix(self):
        """ Converts cbv_array (which is a list of np.ndarrays, one for each
        CBV, to a two-dimensional ndarray where the columns are the CBVs
        """
        return np.array(self.cbv_array).T
 
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
            trim_lc : [bool] If True then also trim the light curve if needed

        """

        if not isinstance(lc, LightCurve):
            raise Exception('<lc> must be a LightCurve class')


        if hasattr(lc, 'cadenceno'):

            lc_trim_mask = np.in1d(lc.cadenceno, self.cadenceno)
            if (np.any(np.logical_not(lc_trim_mask))):
                if (trim_lc):
                    # lc trim method
                    lc.trim(lc_trim_mask)
                else:
                    log.warning('There are cadences in the light curve that are not in the CBVs. NO SYNCHRONIZATION OCCURED')


            trim_mask = np.in1d(self.cadenceno, lc.cadenceno)
            self.cbv_array       = self.cbv_array[:,trim_mask]
            self.cadenceno      = self.cadenceno[trim_mask]
            self.gap_indicators  = self.gap_indicators[trim_mask]

        else:
            log.warning('Synchronization with cadence time stamps is not yet implemented. NO SYNCHRONIZATION OCCURED')

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

    @staticmethod
    def get_kepler_cbv_url(mission, quarter, campaign):
        """ STATIC method to obtain a path to a Kepler/K2 CBV FITS file

            For Kepler extracts the DR25 CBVs

            Gets the html page and finds all references to 'a' tag
            keeps the ones for which 'href' ends with 'fits'
            this might slow things down in case the user wants to fit 1e3 stars
        """
 
        if (mission == 'Kepler'): 
            cbvBaseUrl = "http://archive.stsci.edu/missions/kepler/cbv/"
        elif (mission == 'K2'):
            cbvBaseUrl = "http://archive.stsci.edu/missions/k2/cbv/"
 
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
        return cbvBaseUrl + cbv_file


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

    @staticmethod
    def get_tess_cbv_url(sector, camera, CCD):
        """ STATIC method to obtain a path to a TESS CBV FITS file

            The easiest way to obtain a link to the CBV file for a TESS Sector and camera.CCD is

            1. Download the bulk download curl script (with a predictable url) for the desired sector and search it for the camera.CCD I need
            2. Download the CBV FITS file based on the link in the curl script

            The bulk download curl links have urls such as:

            https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_17_cbv.sh

            Then the individual CBV files foudn in the curl file have urls such as:

            https://archive.stsci.edu/missions/tess/ffi/s0017/2019/279/1-1/tess2019279210107-s0017-1-1-0161-s_cbv.fits

            Returns a url string for the desired CBV FITS file
    
        """
        curlBaseUrl = 'https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_'
        curlEndUrl = '_cbv.sh'
        curlUrl = curlBaseUrl + str(sector) + curlEndUrl

        # This is the string to search for in the curl script file
        # Pad the sector number with a first '0' if less than 10
        # TODO: figure out a way to pad an interger number with forward zeros
        # without needing conditional
        sector = int(sector)
        if (sector < 10):
            curlSearchString = 's000' + str(sector) + '-' + str(camera) + '-' + str(CCD) + '-' 
        elif (sector > 99):
            # We will be blessed if we get to more than 99 sectors!
            raise Exception('Only up to 99 Sectors is currently supported')
        else:
            curlSearchString = 's00' + str(sector) + '-' + str(camera) + '-' + str(CCD) + '-' 

        # 1. Read in the relevent curl script file and find the line for the CBV data we are looking for
        data = urllib.request.urlopen(curlUrl)
        foundIndex = None
        for line in data:
            strLine = str(line)
            try:
                foundIndex = strLine.index(curlSearchString) # str.index will error when not found
                break
            except:
                pass # continue searching
        if (foundIndex is None):
            raise Exception('CBV FITS file not found')

        # extract url from strLine
        htmlStartIndex = strLine.find('https:')
        htmlEndIndex = strLine.rfind('fits')
        cbvUrl = strLine[htmlStartIndex:htmlEndIndex+4] # Add 4 for length of 'fits' string

        return cbvUrl

#*******************************************************************************
# Functions

def get_cbvs (mission=('Kepler', 'K2', 'TESS'), quarter=None, campaign=None,
        sector=None, channel=None, module=None, output=None, camera=None,
        CCD=None, cbv_type='SingleScale', band=None, cbv_indices='ALL'):

    """ Searches the `public data archive at MAST <https://archive.stsci.edu>`
    for Kepler, K2 or TESS cotrending basis vectors.

    This function fetches the Cotrending Basis Vectors FITS HDU for the desired
    mission and channel or CCD, etc... and then extracts the requested basis
    vectors

    For Kepler/K2, the FITS files contain all channels in a single file per
    quarter/campaing. But for TESS, each CCD CBVs are stored in a seperate FITS
    file. So, the retrieval process is slightly different betweem the missions.

    Depending on which mission you are searching for different named parameters
    must be passed as discussed below. If additional named arguments are passed
    then an error is thrown

    Parameters
    ----------
    mission     : str, list of str
                    'Kepler', 'K2', or 'TESS'.
    quarter or campaign or sector : int, list of ints
                    Kepler Quarter, K2 Campaign, or TESS Sector number.
    channel or module and output : int
                    Kepler/K2  requested channel or module and output
                    Must provide either channel, or module and outout
    camera and CCD : int, list of ints
                    TESS camera and CCD
    cbv_type    : str
                    'SingleScale' or 'MultiScale' or 'Spike'
                    For Kepler/K2 only single-scale is available
    band        : int
                    Multi-scale band number
    cbv_indices : int array
                    List of CBVs extracted from FITS file, 1-Based {'ALL' => extract all}

    Returns
    -------
    result : :class:`CBVFile` object
        Object detailing the data products found.

    Examples
    --------
    This example will read in the CBVs for Kepler quarter 8, 
    and then extract the first 8 CBVs for module.output 16.4

        >>> cbvs = get_cbvs(mission='Kepler', quarter=8, module=16, output=4,   # doctest: +SKIP
        >>>     cbv_indices=np.arange(1,9))                                     # doctest: +SKIP

    This example will read in the CBVs for TESS Sector 10 Camera.CCD 2.4
    and then extract the first 6 CBVs of multi-scale band 2

        >>> cbvs = search_cbvs(mission='TESS', sector=10, camera=2, CCD=4, # doctest: +SKIP
        >>>     cbv_type='MultiScale', band=2, cbv_indices=np.arange(1,7)) # doctest: +SKIP
    """

    #***
    # Validate inputs 
    # Make sure only the appropriate arguments are passed
    # TODO: figure out a more elegant way to do this
    if (mission == 'Kepler'):
        assert  isinstance(quarter, int), 'quarter must be passed for Kepler mission'
        # All these inputs are invalid
        assert  campaign is None, 'campaign must not be passed for Kepler mission'
        assert  sector is None,   'sector must not be passed for Kepler mission'
        assert  camera is None,   'camera must not be passed for Kepler mission'
        assert  CCD is None,      'CCD must not be passed for Kepler mission'
    elif (mission == 'K2'):
        assert  isinstance(campaign, int), 'campaign must be passed for K2 mission'
        # All these inputs are invalid
        assert  quarter is None,  'quarter must not be passed for K2 mission'
        assert  sector is None,   'sector must not be passed for K2 mission'
        assert  camera is None,   'camera must not be passed for K2 mission'
        assert  CCD is None,      'CCD must not be passed for K2 mission'
    elif (mission == 'TESS'):
        assert  isinstance(sector, int),    'sector must be passed for TESS mission'
        assert  not camera is None,   'camera must be passed'
        assert  not CCD is None,      'CCD must be passed'
        # All these inputs are invalid
        assert  quarter is  None,  'quarter must not be passed for TESS mission'
        assert  campaign is None, 'campaign must not be passed for TESS mission'
    else:
        raise Exception('Unknown mission type')
        
    try: 

        

        if (mission == 'Kepler' or mission == 'K2'): 

            # CBV FITS files use module/output, not channel
            # So if channel is passed, convert to module/output
            if (isinstance(channel, int)):
                assert  module is None, 'module must NOT be passed if channel is passed'
                assert  output is None, 'output must NOT be passed if channel is passed'
                module, output = channel_to_module_output(channel)
                channel = None
            else:
                assert  not module is None, 'module must be passed'
                assert  not output is None, 'output must be passed'

            kepler_cbv_url = KeplerCotrendingBasisVectors.get_kepler_cbv_url(mission, quarter, campaign)

            hdu = pyfits.open(kepler_cbv_url)

            cbvs = KeplerCotrendingBasisVectors(mission, hdu, module, output,
                    cbv_indices=cbv_indices)

            return cbvs

        else:

            tess_cbv_url = TessCotrendingBasisVectors.get_tess_cbv_url(sector, camera, CCD)

            hdu = pyfits.open(tess_cbv_url)
            
            # Check that this is a TESS CBV FITS file
            mission = hdu['Primary'].header['TELESCOP']
            validate_method(mission, ['tess'])

            cbvs = TessCotrendingBasisVectors(hdu, cbv_type, band, cbv_indices=cbv_indices)

            return cbvs
            

    except:
        raise Exception('CBVS were not found')
        return None


#*******************************************************************************
# Correctors


class CBVCorrector(Corrector):
    """ Class for removing systematics using CBV correctors for Kepler/K2/TESS



    Attributes
    ----------
    lc  : LightCurve
        The light curve to correct
    cbv_type : str list
        List of CBV types to use
    cbv_indices : list of lists
        List of CBV vectors to use in each of cbv_type passed. {'ALL' => Use all}
    cbvs    : CetrendingBasisVectors
        The retrieved CBVs
    cbv_design_matrix : DesignMatrix
        The retrieved CBVs ported into a DesignMatrix object
    extra_design_matrix : DesignMatrix
        An extra design matrix to include in the fit with the CBVs
    design_matrix   : DesignMatrixCollection
        The design matrix collection composed of cbv_design_matrix and extra_design_matrix 
    regression_corrector : RegressionCorrector
        The constrcuted RegressionCorrector to be used to perform the fit
    corrected_lc : LightCurve
        The returned light curve from regression_corrector.correct


    """

    def __init__(self, lc, cbv_type='SingleScale', cbv_indices='ALL', band=None, ext_dm=None):
        """ Constructor for CBVClass objects

        This constructor will retrieve the desired CBVs from MAST and then
        align them with the passed light curve.

        For TESS we have the option to load multiple CBV types
        
        Attributes
        ----------
        lc  : LightCurve
            The light curve to correct
        cbv_type : str list
            List of CBV types to use
        cbv_indices : list of lists
            List of CBV vectors to use in each passed cbv_type. {'ALL' => Use all}
        band        : int list
            Multi-Scale band number to correspond to the passed cbv_type
            For non-Multi-Scale CBVs use None
        ext_dm  : DesignMatrix
            Optionall pass an extra design matrix to also be used in the fit

        """

        if not isinstance(lc, LightCurve):
            raise Exception('<lc> must be a LightCurve class')

        if (isinstance(cbv_type, list)):
            if (not lc.mission == 'TESS'):
                raise Exception('Multiple CBV types are only allowed for TESS')
            if (not len(cbv_type) == len(cbv_indices)):
                raise Exception('cbv_type and cbv_indices must be the same list length')

        self.lc = lc
        self.cbv_type = cbv_type

        # If any DesignMatrix was passed the store it
        self.extra_design_matrix = ext_dm

        #***
        # Retrieve the CBVs from MAST
        # TODO: create CBV collection class
        cbvs = []
        if (isinstance(cbv_indices, str) and (cbv_indices == 'ALL')):
            cbv_indices=np.arange(1,DEFAULT_NUMBER_CBVS+1)

        if self.lc.mission == 'Kepler':
            cbvs.append(get_cbvs(mission=self.lc.mission, quarter=self.lc.quarter, 
                    channel=self.lc.channel, cbv_indices=cbv_indices))
            self.cbv_indices = cbvs.cbv_indices
        elif self.lc.mission == 'K2':
            cbvs.append(get_cbvs(mission=self.lc.mission, campaign=self.lc.campaign, 
                    channel=self.lc.channel, cbv_indices=cbv_indices))
            self.cbv_indices = cbvs.cbv_indices
        elif self.lc.mission == 'TESS':
            # For TESS we can load multiple CBV types
            # TODO: finish this for band input list
            self.cbv_indices = []
            for idx in np.arange(len(cbv_indices)):
                cbvs.append(get_cbvs(mission=self.lc.mission, sector=self.lc.sector,
                    camera=self.lc.camera, CCD=self.lc.ccd, cbv_type=cbv_type[idx],
                    band=band, cbv_indices=cbv_indices[idx]))
                self.cbv_indices.append(cbvs[idx].cbv_indices)
        else:
            raise ValueError('Unknown mission type')

        for idx in np.arange(len(cbvs)):
            if (not isinstance(cbvs[idx], CotrendingBasisVectors)):
                raise Exception('CBVs could not be loaded. CBVCorrector must exit')

        # Align the CBVs with the lightcurve flux using the cadence numbers
        for idx in np.arange(len(cbvs)):
            cbvs[idx].align(self.lc, trim_lc=True)

        self.cbvs = cbvs


    def correct(self):
        """ Performs the correction using RegressionCorrector methods

        This method will assemble the full design matrix collection composed of
        cbv_design_matrix and extra_design_matrix. Then use
        RegressionCorrector.correct to perform the correction

        TODO: A whole bunch! This is a shell of a method. Just to demonstrate
        basis functionality. We should consider allowing for standard
        regaulrization methods such as Ridge Regression (L2 Norm) and Lasso (L1
        Norm).  Maybe even BIC or AIC methods also.

        Eventually, the plan is to use over-fitting and under-fitting goodness
        metrics to constrain the regression fit.


        """

        # Create the design matrix collection with CBVs, plus extra passed basis vectors

        #Create a CBV design matrix for each CBV set loaded
        self.cbv_design_matrix = []
        for idx in np.arange(len(self.cbvs)):
            cbv_index_names = [cbv_index for cbv_index in self.cbv_indices[idx]]
            self.cbv_design_matrix.append(DesignMatrix(self.cbvs[idx]._cbvs_to_matrix(),
                    name=self.cbvs[idx].cbv_type, columns=cbv_index_names))

        # Add the passed in design matrix
        # If none passed then still add the constant offset term (array of ones)
        if (self.extra_design_matrix is None):
            ones_df = pd.DataFrame(np.atleast_2d(np.ones(self.cbv_design_matrix[0].shape[0])).T, columns=['offset'])
            extra_design_matrix = DesignMatrix(ones_df, name='offset')
        else:
            extra_design_matrix = self.extra_design_matrix.append_constant()

        # Create the full design matrix collection
        dm_to_flatten = [[cbv_dm for cbv_dm in self.cbv_design_matrix], [extra_design_matrix]]
        flattened_dm_list = [item for sublist in dm_to_flatten for item in sublist]
        self.design_matrix = DesignMatrixCollection(flattened_dm_list)

            
        # Use RegressionCorrector for the actual fitting
        self.regression_corrector = RegressionCorrector(self.lc)
        self.corrected_lc = self.regression_corrector.correct(self.design_matrix)
        return self.corrected_lc

    def diagnose(self):
        """Returns diagnostic plots to assess the most recent call to `correct()`.

        If `correct()` has not yet been called, a ``ValueError`` will be raised.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """

        if not hasattr(self, 'corrected_lc'):
            raise ValueError('Please call the `correct()` method before trying to diagnose.')

        return self.regression_corrector.diagnose()




#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
# CANDIDATE FOR REMOVAL
# Do we wish to retain any othis functionality? Or is it superseeded by
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
            cbvs = get_cbvs(mission=self.lc.mission, quarter=self.lc.quarter, 
                    channel=self.lc.channel, cbv_indices='ALL')
        elif self.lc.mission == 'K2':
            cbvs = get_cbvs(mission=self.lc.mission, campaign=self.lc.campaign, 
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

