"""
Defines the CBVFile classes.

These are used to retrieve Cotrending Basis Vectors (CBVs) from the archive.

The CBVs are then to be incoroporated into a CotrendingBasisVectors object before use to correct light curves.

"""

import logging
import requests
import numpy as np
from astropy.io import fits as pyfits
from bs4 import BeautifulSoup
from ..utils import channel_to_module_output
from .CBV import KeplerCotrendingBasisVectors

__all__ = ['KeplerCBVFile', 'TessCBVFile']

log = logging.getLogger(__name__)

class CBVFile(object):
    """ Generic class for storing the data from CBV files from the archive

    """

    #***
    # Some constants
    @property
    def nCBVsDefault(self):
        """ Default number of CBVs """
        return 16

    #***
    def __init__(self, path, **kwargs):
        """ Contructor: Fetches a CBV FITS file from the archive 

        Parameters
        ----------
        path : str or `astropy.io.fits.HDUList` object
            Local path or remote url of a lightcurve FITS file.
            Also accepts a FITS file object already opened using AstroPy.
        kwargs : dict
            Keyword arguments to be passed to astropy.io.fits.open.
        """
        if isinstance(path, pyfits.HDUList):
            self.path = None
            self.hdu = path
        else:
            self.path = path
            self.hdu = pyfits.open(self.path, **kwargs)

  # @property
  # def mission(self):
  #     """ Curiously, Kepler/K2 and TESS CBV FITS files does not have a "MISSION" keyword 
  #         We must define the mission in the subclass
  #     """

  #     log.error('Mission not defined')

  # @mission.setter
  # def mission(self, name):
  #     self.mission = name

    def header(self, ext=0):
        """Header of the object at extension `ext`"""
        return self.hdu[ext].header

    def get_keyword(self, keyword, ext=0, default=None):
        """Returns a header keyword value.

        If the keyword is Undefined or does not exist,
        then return ``default`` instead.
        """
        try:
            kw = self.hdu[ext].header[keyword]
        except KeyError:
            return default
        if isinstance(kw, Undefined):
            return default
        return kw

    def get_cbvs (self, cbvType='singleScale', cbvIndices='ALL'):

        pass


        
class KeplerCBVFile(CBVFile):
    """ Subclass for Kepler CBVs
 
         Only the singleScale CBVs are archived for Kepler
 
    """
 
    def __init__ (self, path, mission=None):
        """ Constructor:

            There is no "MISSION" keyword in the CBV FITS files so that must be explicitely given
        """

        super(KeplerCBVFile, self).__init__(path)
        
        assert not mission is None, log.error('"mission" must be passed for Kepler/K2 CBV Fits files')
        self.mission = mission

        pass

  # @property
  # def mission(self):
  #     """'Kepler' or 'K2'"""
  #     return self.mission

  # @mission.setter
  # def mission(self, name):
  #     self.mission = name

    def get_cbvs(self, channel=None, module=None, output=None, cbvIndices='ALL'):
        """ Returns the requested CBVs as a CotrendingBasisVectors object

            Input:
                channel     -- [int] If passed then module and output must NOT be
                module      -- [int] If passed then channel must Not be
                output      -- [int] If passed then channel must Not be
                cbvIndices  -- [int array] List of CBVs to get (does not need to be in sequential order, and can skip indices)
        """

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


        cbvs = KeplerCotrendingBasisVectors(self.mission, self.hdu, module, output, cbvIndices=cbvIndices)
 
        return cbvs


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

    get_kepler_cbv_url = staticmethod(get_kepler_cbv_url)

