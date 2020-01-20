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
            log.error('Unknown mission')

        # Check if a valid cbvType was passed
        try:
            self.validCBVTypes.index(cbvType)
            self.cbvType = cbvType
        except:
            log.error('Unknown mission')

        self.band = band

        # For Kepler/K2/TESS it's always been 16 CBVs
        # TODO: automate this based on the CBVs in the FITS file, to future-proof potential changes
        if (cbvIndices == 'ALL'):
            cbvIndices=np.arange(1,17)
        self.cbvIndices = cbvIndices
 
        

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

        cbv_data = HDU['MODOUT_{0}_{1}'.format(module, output)].data

        self.cbvCadenceNo    = HDU['MODOUT_{0}_{1}'.format(module, output)].data['CADENCENO']
        self.gapIndicators   = HDU['MODOUT_{0}_{1}'.format(module, output)].data['GAPFLAG']
        self.cbvExtName      = HDU['MODOUT_{0}_{1}'.format(module, output)].header['EXTNAME']

        cbvArray = []
        for i in self.cbvIndices:
            cbvArray.append(cbv_data.field('VECTOR_{}'.format(i)))
        self.cbvArray = np.asarray(cbvArray)

