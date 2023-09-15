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
import numpy as np
import scipy
from ..utils import channel_to_module_output, plot_image
from astropy.io import fits as pyfits
import math

#from ..targetpixelfile import TargetPixelFile


# Just sketching out a few models right now
class PRF(ABC):

	@abstractmethod
	def __init__(self, column, row):
		"""
		A generic base class object for PRFs You're not supposed to use this.
		See KeplerPRF and TessPRF for the instantiable classes. 
		
		column = pixel coordinate of the lower left column value
		column = pixel coordinate of the lower left row value
		"""
		
		# load PSF file
		# Initialize object
		self.column = column
		self.row = row
		
	def __repr__(self):
		return "I'm an abstract PRF"
		
	def __call__(self):
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
		
	def evaluate(self,
		center_col = None, # If not specified, make output 10x10 with prf in center?
		center_row = None, # if not provided, use corner_row + self.shape[1] / 2
		flux=1.0,
		scale_col=1.0,
		scale_row=1.0,
		rotation_angle=0.0,):
		
		"""
		Interpolates the PRF model onto detector coordinates.

		Parameters
		----------
		center_col, center_row : float
			Column and row coordinates of the center
		flux : float
			Total integrated flux of the PRF
		scale_col, scale_row : float
			Pixel scale stretch parameter, i.e. the numbers by which the PRF
			model needs to be multiplied in the column and row directions to
			account for focus changes
		rotation_angle : float
			Rotation angle in radians

		Returns
		-------
		prf_model : 2D array
			Two dimensional array representing the PRF values parametrized
			by flux, centroids, widths, and rotation as applicble.
		"""
		if center_col == None:
			center_col=self.column + self.shape[1] / 2
		if center_row == None:
			center_row=self.row+ self.shape[0] / 2
		
		if ((scale_col == 1.0) and (scale_row == 1.0) and (rotation_angle == 0.0)):
			delta_col = self.col_coord - center_col
			delta_row = self.row_coord - center_row
			self.prf_model = flux * self.interpolate(delta_row, delta_col)
		
		else:
			cosa = math.cos(rotation_angle)
			sina = math.sin(rotation_angle)

			delta_col = self.col_coord - center_col
			delta_row = self.row_coord - center_row
			delta_col, delta_row = np.meshgrid(delta_col, delta_row)

			rot_row = delta_row * cosa - delta_col * sina
			rot_col = delta_row * sina + delta_col * cosa

			self.prf_model = flux * self.interpolate(
				rot_row.flatten() * scale_row, rot_col.flatten() * scale_col, grid=False
			).reshape(self.shape)

		
		# CUTOUT THE SIZE OF THE PRF MODEL
		# Rect bivariate spline extrapolates when the grid goes beyond the original coordinates
		#	  resulting in trailing col/rows of low values. We don't need to deal with that. 
		#print(f"testcol: {delta_col}")
		#print(f"testro: {delta_row}")
		#testcol = np.abs(delta_col) >= 6.5
		#testrow = np.abs(delta_row) >= 6.5
		
		
		#cutout_prf = self.prf_model.copy()
		#cutout_prf[testcol] = 0
		#cutout_prf[testrow] = 0
		#self.prf = cutout_prf

		return self.prf_model
			
		
		
	@abstractmethod
	def _prepare_prf(self):
		pass

	@abstractmethod
	def _read_prf_calibration_file(self):
		pass 

		
#############################################################		
#############################################################
#						KEPLER PRF
#############################################################
#############################################################
class KeplerPRF(PRF):
	"""A KeplerPRF class.
	
	There are 5 PRF measurements (the 4 corners and the center) for each channel.
	The measured PRF is over-sampled by a factor of 50 to allow for sub-pixel interpolation.
	The model is a 550x550 (or 750x750) grid that covers 11x11 (or 15x15) pixels
	"""
	# I want the option to either give it a tpf and it reads channel/shape OR provide that info
	def __init__(self, column, row, channel, shape):
		super().__init__(column=column, row=row)
		self.channel = channel
		self.shape = shape
		(
			self.col_coord,
			self.row_coord,
			self.interpolate,
			self.supersampled_prf,
		) = self._prepare_prf()
		
	def __repr__(self):
		return "I'm a Kepler PRF"
		
	def __call__(self, column, row, channel, shape, **kwargs): # Add more here
		return self.evaluate(
			center_col, center_row, **kwargs
		)
	
	def evaluate(self, **kwargs):
		'''
		Optional keywords:
		center_col (default is the center of the image)
		center_row (default is the center of the image)
		flux (default 1.0),
		scale_col (default 1.0, ie no scaling),
		scale_row (default 1.0, ie no scaling),
		rotation_angle (default 0.0, ie no rotation)
		'''
		return super().evaluate(**kwargs) # Make kwargs explicit?
		
	def _read_prf_calibration_file(self, path, ext):
		prf_cal_file = pyfits.open(path)
		data = prf_cal_file[ext].data
		crval1p = prf_cal_file[ext].header["CRVAL1P"]
		crval2p = prf_cal_file[ext].header["CRVAL2P"]
		cdelt1p = prf_cal_file[ext].header["CDELT1P"]
		cdelt2p = prf_cal_file[ext].header["CDELT2P"]
		prf_cal_file.close()

		return data, crval1p, crval2p, cdelt1p, cdelt2p
		
	def _prepare_prf(self):
		n_hdu = 5 # measurements at the 4 corners + center
		min_prf_weight = 1e-6
		module, output = channel_to_module_output(self.channel)
		# determine suitable PRF calibration file
		module = str(module).zfill(2)
		prfs_url_path = "http://archive.stsci.edu/missions/kepler/fpc/prf/kplr"
		prffile = (
			prfs_url_path
			+ str(module)
			+ "."
			+ str(output)
			+ "_2011265_prf.fits"
		)
		print(prffile)

		# read PRF images
		prfn = [0] * n_hdu
		crval1p = np.zeros(n_hdu, dtype="float32")
		crval2p = np.zeros(n_hdu, dtype="float32")
		cdelt1p = np.zeros(n_hdu, dtype="float32")
		cdelt2p = np.zeros(n_hdu, dtype="float32")

		for i in range(n_hdu):
			(
				prfn[i],
				crval1p[i],
				crval2p[i],
				cdelt1p[i],
				cdelt2p[i],
			) = self._read_prf_calibration_file(prffile, i + 1)

		prfn = np.array(prfn) 
		PRFcol = np.arange(0.5, np.shape(prfn[0])[1] + 0.5)
		PRFrow = np.arange(0.5, np.shape(prfn[0])[0] + 0.5)
		# Shifts pixels so it is in pixel units centered on 0
		PRFcol = (PRFcol - np.size(PRFcol) / 2) * cdelt1p[0]
		PRFrow = (PRFrow - np.size(PRFrow) / 2) * cdelt2p[0]

		# interpolate the calibrated PRF shape to the target position
		rowdim, coldim = self.shape[0], self.shape[1]
		supersamp_prf = np.zeros(np.shape(prfn[0]), dtype="float32")
		ref_column = self.column + 0.5 * coldim
		ref_row = self.row + 0.5 * rowdim

		for i in range(n_hdu):
			prf_weight = math.sqrt(
				(ref_column - crval1p[i]) ** 2 + (ref_row - crval2p[i]) ** 2
			)
			if prf_weight < min_prf_weight:
				prf_weight = min_prf_weight
			supersamp_prf += prfn[i] / prf_weight

		supersamp_prf /= np.nansum(supersamp_prf) * cdelt1p[0] * cdelt2p[0]
		# location of the data image centered on the PRF image (in PRF pixel units)
		col_coord = np.arange(self.column + 0.5, self.column + coldim + 0.5)
		row_coord = np.arange(self.row + 0.5, self.row + rowdim + 0.5)
		# x-axis correspond to row-axis in scipy.RectBivariate
		# not to be confused with our convention, in which the
		# x-axis correspond to the column-axis
		interpolate = scipy.interpolate.RectBivariateSpline(PRFrow, PRFcol, supersamp_prf)
		
		return col_coord, row_coord, interpolate, supersamp_prf
	
	def from_tpf(self):
		'''Creates a PRF object using a TPF'''
		# Add some error checks that it's a Kepler TPF here
		print(self.shape)
		test_prf =  KeplerPRF(self.column, self.row, self.channel, self.shape[1:3])
		print(test_prf)
		return test_prf.evaluate()
		
	def plot(self, *params, **kwargs): # Fill in params explicitly
		#pflux = self.evaluate(*params) # Check if there is already a PRF model, and if not evaluate
		plot_image(
			pflux,
			title=f"Kepler PRF Model, Channel: {self.channel}, CCD: {self.ccd}",
			extent=(
				self.column,
				self.column + self.shape[1],
				self.row,
				self.row + self.shape[0],
			),
			**kwargs
		)	

		
#############################################################		
#############################################################
#						 TESS PRF
#############################################################
#############################################################
class TessPRF(PRF):
	"""A TessPRF class""" 
	def __init__(self, column, row, camera, ccd, sector):
		# Sector needed as different engineering files are needed for sectors < 4
		super().__init__(column=column, row=row)
		self.camera = camera
		self.ccd = ccd
		self.sector = sector
		
	def __repr__(self):
		return "I'm a TESS PRF"
		
	def __call__(self, center_col, center_row, **kwargs):
		return self.model(
			center_col, center_row, **kwargs
		)
	
	def _read_prf_calibration_file(self, path):
		# The TESS fits file is different than Kepler. We always want the first extension. 
		# The second is the errors.
		# TODO: These files do not crop out noisy data, unlike Kepler that zeros out the fringe. 
		#	   Remake these and save them out locally. 
		prf_cal_file = fits.open(path)
		
		data = prf_cal_file[0].data
		crval1p = prf_cal_file[0].header["CRVAL1P"]
		crval2p = prf_cal_file[0].header["CRVAL2P"]
		cdelt1p = prf_cal_file[0].header["CDELT1P"]
		cdelt2p = prf_cal_file[0].header["CDELT2P"]
		prf_cal_file.close()

		return data, crval1p, crval2p, cdelt1p, cdelt2p	
		
	def _prepare_prf(self):
		print(f"preparing PRF: {self.camera},{self.ccd}")
		min_prf_weight = 1e-6 

		# TESS has a separate file for each point in a grid of pixel locations.
		# Find the closest 4 to your pixel location and go from there.
		cols= np.array([45,557,1069,1580,2092,45,557,1069,1580,2092,45,557,
			1069,1580,2092,45,557,1069,1580,2092,45,557,1069,1580,2092])
		rows= np.array([1,1,1,1,1,513,513,513,513,513,1025,1025,1025,1025,
			1025,1536,1536,1536,1536,1536,2048,2048,2048,2048,2048])
		# I think this simplifies things a little bit
		pythagorus = np.sqrt((rows-self.row)**2. + (cols-self.column)**2.)
		# Provide the index for the row/column combination that make up a box around the target location
		four_corners = np.argpartition(pythagorus,4)[:4] 
		
		
		# Not all ccd/cams have the same date, so need to do a bit of finagling
		# NOTE IF I REMAKE THESE FILES, I WILL DROP THIS DATA TO SIMPLIFY
		if self.sector < 4:
			prfSector = 1
			prfs_url_path = "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/start_s0001/"
			if (self.camera == 1):
				prffiles = [f"{prfs_url_path}cam{self.camera}_ccd{self.ccd}/tess2018243163600-prf-{self.camera}-{self.ccd}-row{rows[c]:04}-col{cols[c]:04}.fits" for c in four_corners]
			elif ((self.camera == 2) & (self.ccd <= 3)):
				prffiles = [f"{prfs_url_path}cam{self.camera}_ccd{self.ccd}/tess2018243163600-prf-{self.camera}-{self.ccd}-row{rows[c]:04}-col{cols[c]:04}.fits" for c in four_corners]
			else:
				prffiles = [f"{prfs_url_path}cam{self.camera}_ccd{self.ccd}/tess2018243163601-prf-{self.camera}-{self.ccd}-row{rows[c]:04}-col{cols[c]:04}.fits" for c in four_corners]
		else:
			prfSector = 4
			prfs_url_path = "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/start_s0004/"
			if (self.camera == 1) & (self.ccd == 1):
				prffiles = [f"{prfs_url_path}cam{self.camera}_ccd{self.ccd}/tess2019107181900-prf-{self.camera}-{self.ccd}-row{rows[c]:04}-col{cols[c]:04}.fits" for c in four_corners]
			elif ((self.camera == 1) & (self.ccd > 1)) \
				| (self.camera == 2) \
				| ((self.camera == 3) & (self.ccd == 1)):
				prffiles = [f"{prfs_url_path}cam{self.camera}_ccd{self.ccd}/tess2019107181901-prf-{self.camera}-{self.ccd}-row{rows[c]:04}-col{cols[c]:04}.fits" for c in four_corners]
			else:
				prffiles = [f"{prfs_url_path}cam{self.camera}_ccd{self.ccd}/tess2019107181902-prf-{self.camera}-{self.ccd}-row{rows[c]:04}-col{cols[c]:04}.fits" for c in four_corners]
		
		# Read PRF images
		# This format is to stay consistent with how Kepler's files are read in
		n_hdu = len(four_corners) 
		prfn = [0] * n_hdu
		crval1p = np.zeros(n_hdu, dtype="float32")
		crval2p = np.zeros(n_hdu, dtype="float32")
		cdelt1p = np.zeros(n_hdu, dtype="float32")
		cdelt2p = np.zeros(n_hdu, dtype="float32")
	
		for i in range(n_hdu):
			(
				prfn[i],
				crval1p[i],
				crval2p[i],
				cdelt1p[i],
				cdelt2p[i],
			) = self._read_prf_calibration_file(prffiles[i])
		
		prfn = np.array(prfn)
		PRFcol = np.arange(0.5, np.shape(prfn[0])[1] + 0.5)
		PRFrow = np.arange(0.5, np.shape(prfn[0])[0] + 0.5)
		PRFcol = (PRFcol - np.size(PRFcol) / 2) * cdelt1p[0]
		PRFrow = (PRFrow - np.size(PRFrow) / 2) * cdelt2p[0]

		# interpolate the calibrated PRF shape to the target position
		rowdim, coldim = self.shape[0], self.shape[1]
		supersamp_prf = np.zeros(np.shape(prfn[0]), dtype="float32")
		ref_column = self.column + 0.5 * coldim
		ref_row = self.row + 0.5 * rowdim

		for i in range(n_hdu):
			prf_weight = math.sqrt(
				(ref_column - crval1p[i]) ** 2 + (ref_row - crval2p[i]) ** 2
			)
			if prf_weight < min_prf_weight:
				prf_weight = min_prf_weight
			supersamp_prf += prfn[i] / prf_weight
			
		supersamp_prf /= (np.nansum(supersamp_prf) * cdelt1p[0] * cdelt2p[0])
		
		# location of the data image centered on the PRF image (in PRF pixel units)
		# NOTE IN STEVE'S CODE, HE ADDS 1 TO SELF.COLUMN/SELF.ROW AND COLDIM/ROWDIM
		col_coord = np.arange(self.column + 0.5, self.column + coldim + 0.5)
		row_coord = np.arange(self.row + 0.5, self.row + rowdim + 0.5)
		# x-axis correspond to row-axis in scipy.RectBivariate
		# not to be confused with our convention, in which the
		# x-axis correspond to the column-axis

		interpolate = scipy.interpolate.RectBivariateSpline(PRFrow, PRFcol, prf)
			
		 
		return col_coord, row_coord, interpolate, prf

	def from_tpf(self):
		'''Creates a PRF object using a TPF'''
		# Add some error checks that it's a Tess TPF here
		return TessPRF(self.column, self.row, self.camera, self.ccd, self.sector)
		
	def plot(self, *params, **kwargs): # Fill in params explicitly
		#pflux = self.evaluate(*params) # Check if there is already a PRF model, and if not evaluate
		plot_image(
			pflux,
			title=f"TESS PRF Model, Sector: {self.sector}, Camera: {self.camera}, CCD: {self.ccd}",
			extent=(
				self.column,
				self.column + self.shape[1],
				self.row,
				self.row + self.shape[0],
			),
			**kwargs
		)

		
		





















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
