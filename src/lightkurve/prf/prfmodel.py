''' 
THIS IS A WORK IN PROGRESS


This PR is an update to prfmodel.py.
In addition to adding functionality for TESS, this PR makes a number of functional and structural changes.

Changes from original

- Adds functionality for TESS
- TESS PRF files are modified to '0-out' pixels that do not contribute to the PRF
- Kepler and TESS files are stored in the lightkurve file structure (ie not downloaded from the internet)
- The Interpolated PRF is cropped so that the flux contribution cannot extend past the PRF file size.
- The structure of the call is changed (specifics of which I'm still deciding on)


Use cases and functionality I am considering:

"I want to make an aperture for a TESSCut TPF"
	- tpf.prf.estimate_aperture()
	
"I want to build a PRF model to simulate a real TPF"
	- Each TargetPixelFile has a prf attribute which is a callable PRF instance tpf.prf. 
	  Users can e.g. do tpf.prf.estimate_aperture() or tpf.prf.estimate_aperture(completeness=0.9) or tpf.prf.model(position)
	- obtain a table of star positions (either in coords, pixel coords, magnitudes or flux)
	- return a np.ndarray of values that can be multiplied by flux summed to create an image.
	
	
"I want to just see what the PRF looks like at a given location"
	- lk.prf.TessPRF(camera=1, ccd=1).plot(loc)
	- lk.prf.TessPRF(camera=1, ccd=1).model(loc)
	- should return a PRF the size of the engineering files (11x11 pixels I believe) by default
	
	
New Workflow:

- Instantiate PRF with only the channel (Kepler) or camera/ccd (TESS)
- Call a 'model' function with the column, row, flux, center_col, center_row, flux, scale_col, scale_row, rotation_angle

'''
from abc import ABC, abstractmethod
from typing import Union
from astropy.coordinates import SkyCoord
import numpy.typing as npt

#from ..targetpixelfile import TargetPixelFile


# Just sketching out a few models right now
class PRF(ABC):
	def __init__(self, fname, pixel_size):
		# load PSF file
		# Initialize object
		raise NotImplementedError
	def estimate_aperture(self, tpf): #tpf: TargetPixelFile) -> npt.ArrayLike:
		# Given a tpf, build a prf model and estimate the best aperture
		# Add in completeness and contamination values (flfrcsap/crowdsap)
		raise NotImplementedError
	def estimate_pipeline_aperture(self, tpf):#tpf: TargetPixelFile) -> npt.ArrayLike:
		# Given a tpf, build a prf model that is close to what the pipeline does
		# Probably a wrapper for the above function with a defined completeness/contamination
		raise NotImplementedError
	def create_simple_aperture(self, coord:Union[tuple, SkyCoord], completeness: float = 0.9, oversample: int = 5) -> npt.ArrayLike:
		# Based on completeness requirement, create an aperture
		# This wouldn't worry about contamination
		# Can take either RA/Dec or pixel coordinates
		raise NotImplementedError
	def create_highres_model(self, oversample: int = 5):
		# make a PSF model on a higher resolution grid in order to estimate the completeness or contamination. 
		# You might have 5x the pixel size as a sane default
		raise NotImplementedError
	def model(self):#,
		'''corner_col,
		corner_row,
		center_col, # if not provided, use corner_col + self.shape[0] / 2
		center_row, # if not provided, use corner_row + self.shape[1] / 2
		flux=1.0,
		scale_col=1.0,
		scale_row=1.0,
		rotation_angle=0.0,):'''
		raise NotImplementedError
		


		
#############################################################		
#############################################################
#						KEPLER PRF
#############################################################
#############################################################
class KeplerPRF(PRF):
	"""A KeplerPRF class"""
	# I want the option to either give it a tpf and it reads channel/shape OR provide that info
	def __init__(self, channel, shape):
		super().__init__(fname=fname, pixel_size=pixel_size)
		self.channel = channel
		self.shape = shape
		
	def __repr__(self):
		return "I'm a Kepler PRF"
		
	def __call__(self, center_col, center_row): # Add more here
		return self.evaluate(
			center_col, center_row, **kwargs
		)
	@staticmethod
	def from_tpf(self):
		'''Creates a PRF object using a TPF'''
		# Add some error checks that it's a Kepler TPF here
		return PRF(tpf.channel, tpf.shape)

		
#############################################################		
#############################################################
#						 TESS PRF
#############################################################
#############################################################
class TessPRF(PRF):
	"""A TessPRF class""" 
	def __init__(self, camera, ccd, shape, sector):
		# Sector needed as different engineering files are needed for sectors < 4
		super().__init__(fname=fname, pixel_size=pixel_size)
		self.camera = camera
		self.ccd = ccd
		self.shape = shape
		self.sector = sector
		
	def __repr__(self):
		return "I'm a TESS PRF"
		
	def __call__(self, center_col, center_row, **kwargs):
		return self.evaluate(
			center_col, center_row, **kwargs
		)
	@staticmethod
	def from_tpf(self, tpf):
		'''Creates a PRF object using a TPF'''
		# Add some error checks that it's a Kepler TPF here
		return PRF(tpf.camera, tpf.ccd, tpf.shape)
		


		
		





















################################################################################
#								   KEPLER  
################################################################################
	def _read_prf_calibration_file(self, path, ext):
		prf_cal_file = pyfits.open(path)
		data = prf_cal_file[ext].data
		#data[data == 0.] = np.nan
		# looks like these data below are the same for all prf calibration files
		crval1p = prf_cal_file[ext].header["CRVAL1P"]
		crval2p = prf_cal_file[ext].header["CRVAL2P"]
		cdelt1p = prf_cal_file[ext].header["CDELT1P"]
		cdelt2p = prf_cal_file[ext].header["CDELT2P"]
		prf_cal_file.close()

		return data, crval1p, crval2p, cdelt1p, cdelt2p
		
		
		
################################################################################
#								   TESS  
################################################################################		
	def _read_prf_calibration_file(self, path):
		# The TESS fits file is different than Kepler. We always want the first extension. 
		# The second is the errors.
		# NOTE: This format will change when I save the cropped images locally
		prf_cal_file = fits.open(path)
		data = prf_cal_file[0].data
		# looks like these data below are the same for all prf calibration files
		crval1p = prf_cal_file[0].header["CRVAL1P"]
		crval2p = prf_cal_file[0].header["CRVAL2P"]
		cdelt1p = prf_cal_file[0].header["CDELT1P"]
		cdelt2p = prf_cal_file[0].header["CDELT2P"]
		prf_cal_file.close()

		return data, crval1p, crval2p, cdelt1p, cdelt2p
