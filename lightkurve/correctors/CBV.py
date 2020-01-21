"""
Defines a CotrendingBasisVectors class.

Stores Cotrending Basis Vectors for the Kepler/K2/TESS missions. 

Use search_cbvs to find the appropriate FITs files and KeplerCBVFile and TessCBVFile to retrieve the CBV FITS files.

Each CotrendingBasisVectors objects contains only ONE set of CBVs. Instantiate multiple objects to stroe multiple set of
CBVS, for example to save all three multi-scale bands.

"""

import logging
import numpy as np

__all__ = ['KeplerCotrendingBasisVectors']

log = logging.getLogger(__name__)

"""
Class Attributes:
    mission         -- [str] ('Kepler', 'K2', 'TESS')
    quarter         -- only for Kepler mission
    campaign        -- only for K2 mission
    sector          -- only for TESS mission
    module,output   -- only for Kepler/K2
    camera,CCD      -- only for TESS
    cbvType         -- [str ('SingleScale', 'MultiScale', 'Spike')
    band            -- [int] MultiScale band number (invalid for other CBV types)
    cbvIndices      -- [int arry] List of CBVs extracted for FITS file, {'ALL' => extract all}
    cbvArray        -- [np.ndarray] The basis vectors
    cbvCadenceNo    -- [int list] Candece indices  
    gapIndicators   -- [bool list]
    cbvExtName      -- [str] The EXTNAME extension in the header

"""

class CotrendingBasisVectors:
    """ Super class for generic components of CBVs
    """

    #***
    # Some constants
    @property
    def validMissionOptions(self):
        return ('Kepler', 'K2', 'TESS')

    @property
    def validCBVTypes(self):
        return ('SingleScale', 'MultiScale', 'Spike')
    #***

    def __init__(self, mission, cbvType='SingleScale', band=None, cbvIndices='ALL'):

        # Check if a valid mission was passed
        try:
            self.validMissionOptions.index(mission)
            self.mission = mission
        except:
            log.error('Invalid mission')

        # Check if a valid cbvType was passed
        try:
            self.validCBVTypes.index(cbvType)
            self.cbvType = cbvType
        except:
            log.error('Invalid cbvType')

        self.band = band

        # For Kepler/K2/TESS it's always been 16 CBVs
        # TODO: automate this based on the CBVs in the FITS file, to future-proof potential changes
        if (isinstance(cbvIndices, str)):
            if (cbvIndices == 'ALL'):
                # TODO: figure out how to do this with one conditional, such as with the && logical in Matlab
                cbvIndices=np.arange(1,17)
        self.cbvIndices = cbvIndices
 
        

#***
class KeplerCotrendingBasisVectors(CotrendingBasisVectors):

    #***
    # Some constants
    @property
    def validMissionOptions(self):
        return ('Kepler', 'K2')

    @property
    def validCBVTypes(self):
        return ('SingleScale')
    #***

    def __init__(self, mission, HDU, module, output, cbvIndices='ALL'):
        """ Kepler/K2 CBVs are all in the same FITS file for each quarter/campaign, so, when intantiating the CBV object
        we must specify which module and output we desire. Only Single-Scale CBVs are stored for Kepler.
        """

        super(KeplerCotrendingBasisVectors, self).__init__(mission, cbvType='SingleScale', band=None, cbvIndices=cbvIndices)
        del mission, cbvIndices # Force use of object attributes

        if (mission == 'Kepler'):
            self.quarter = HDU['Primary'].header['QUARTER']
            self.campaign = None
        elif (mission == 'TESS'):
            self.campaign = HDU['Primary'].header['QUARTER']
            self.quarter = None

        self.module = module
        self.output = output

        extName = 'MODOUT_{0}_{1}'.format(module, output)
        cbv_data = HDU[extName].data

        self.cbvCadenceNo    = HDU[extName].data['CADENCENO']
        self.gapIndicators   = HDU[extName].data['GAPFLAG']
        self.cbvExtName      = HDU[extName].header['EXTNAME']

        cbvArray = []
        for i in self.cbvIndices:
            cbvArray.append(cbv_data.field('VECTOR_{}'.format(i)))
        self.cbvArray = np.asarray(cbvArray)

#***
class TessCotrendingBasisVectors(CotrendingBasisVectors):

    #***
    # Some constants
    @property
    def validMissionOptions(self):
        return ('TESS')

    @property
    def validCBVTypes(self):
        return ('SingleScale', 'MultiScale', 'Spike')
    #***

    def __init__(self, HDU, cbvType, band, cbvIndices='ALL'):
        """ TESS CBVs are in seperate FITS files for each camera.CCD, so camera.CCD is already specified in the
        TessCBVFile object, here we need to specify which CBV type and band is desired.
        """

        mission = HDU['PRIMARY'].header['TELESCOP']
        assert mission == 'TESS', log.error('This does not appear to be a TESS FITS HDU')

        super(TessCotrendingBasisVectors, self).__init__(mission, cbvType=cbvType, band=band, cbvIndices=cbvIndices)
        del mission, cbvType, band, cbvIndices # Force use of object attributes

        self.sector = HDU['PRIMARY'].header['SECTOR']
        # Curiosly, camera and CCD are not in the primary header!
        self.camera = HDU[1].header['CAMERA']
        self.CCD = HDU[1].header['CCD']

        # Get the requested cbvType
        switcher = {
            'SingleScale': 'CBV.single-scale.{}.{}'.format(self.camera, self.CCD),
            'MultiScale': 'CBV.multiscale-band-{}.{}.{}'.format(self.band, self.camera, self.CCD),
            'Spike': 'CBV.spike.{}.{}',
            'unknown': 'error'
            }
        extName = switcher.get(self.cbvType, switcher['unknown'])
        if (extName == 'error'):
            log.error('Invalide cbvType')


        cbv_data = HDU[extName].data

        self.cbvCadenceNo    = HDU[extName].data['CADENCENO']
        self.gapIndicators   = HDU[extName].data['GAP']
        self.cbvExtName      = HDU[extName].header['EXTNAME']

        cbvArray = []
        for i in self.cbvIndices:
            cbvArray.append(cbv_data.field('VECTOR_{}'.format(i)))
        self.cbvArray = np.asarray(cbvArray)

