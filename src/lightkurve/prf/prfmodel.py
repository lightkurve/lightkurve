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
	  Users can e.g. do tpf.prf.estimate_aperture() or tpf.prf.estimate_aperture(completeness=0.9) or tpf.prf.evaluate(position)
	- obtain a table of star positions (either in coords, pixel coords, magnitudes or flux)
	- return a np.ndarray of values that can be multiplied by flux summed to create an image.
	
	
"I want to just see what the PRF looks like at a given location"
	- lk.prf.TessPRF(camera=1, ccd=1).plot(loc)
	- lk.prf.TessPRF(camera=1, ccd=1).evaluate(loc)
	- should return a PRF the size of the engineering files (11x11 pixels I believe) by default
	
	
New Workflow:

- Instantiate PRF with only the channel (Kepler) or camera/ccd (TESS)
- Call a 'model' function with the column, row, flux, center_col, center_row, flux, scale_col, scale_row, rotation_angle

Question: Do we want to preserve the gradient function?

'''
from abc import ABC, abstractmethod
from typing import Union
from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename
import numpy.typing as npt
import numpy as np
import scipy
from ..utils import channel_to_module_output, plot_image
from astropy.io import fits as pyfits
import math
import matplotlib.pyplot as plt
from matplotlib import patches
#from ..targetpixelfile import _parse_aperture_mask



class PRF(ABC):

	@abstractmethod
	def __init__(self, column:int, row:int, shape:tuple = (11,11)):
		"""
		A generic base class object for PRFs You're not supposed to use this.
		See KeplerPRF and TessPRF for the instantiable classes. 
		
		Parameters
		----------
		column : int
		 	pixel coordinate of the lower left column value
		row : int
		 	 pixel coordinate of the lower left row value
		shape : tuple
			shape of any resultant PRFs
		"""
		
		# load PSF file
		# Initialize object
		self.column = column
		self.row = row
		self.shape = shape
		
	def __repr__(self):
		return "I'm an abstract PRF"
		
	# def __call__(self):
	# 	raise NotImplementedError
		
		
	'''def estimate_pipeline_aperture(self, tpf: TargetPixelFile) -> npt.ArrayLike:
		# Given a tpf, build a prf model that is close to what the pipeline does
		# Probably a wrapper for the above function with a defined completeness/contamination
		raise NotImplementedError'''
		
	def get_simple_aperture(self, prf_model: npt.ArrayLike, idx: int = 0, min_completeness: float = 0.9, plot: bool=False) -> npt.ArrayLike: 
		'''
		Based on completeness requirement, create an aperture.
		This simple aperture does NOT account for contamination by other sources.
		
		Parameters:
		-----------
		prf_model : npt.ArrayLike
			3D cube of the PRFs of all sources with shape (nsources x nrows x ncolumns)
			Only the target source is used for this function. 
		idx : int
			The index of the source to be considered as the target (default 0)
		min_completeness : float
			Minimum fraction of flux contained within the aperture
		plot : bool
			plots the simple aperture over the model PRF
		
		Returns:
		--------
		aperture : npt.ArrayLike 
			2D boolean array of the same (nrows x ncolumns) as `prf_model`.
			True where source is inside aperture
		'''
		prf = prf_model[idx, :, :]

		if np.sum(prf) > 1.01: # The PRF occasionally sums to slightly more
			raise ValueError("`prf_model` must sum to 1 or less.")
		
		# If completeness = 1, return any pixel that contains any amount of flux
		if min_completeness == 1.0:
			aperture = prf.astype(bool) 
			aperture[prf != 0.0] = True

		else:
			sort = np.argsort(prf.flatten())
			cusu = np.cumsum(prf.flatten()[sort])
			
			# indices that are "inside" aperture
			ap_index = sort[(1-cusu) < min_completeness]
			aperture = np.zeros(prf.shape, dtype=bool).flatten()
			aperture[ap_index] = True
			# reshape to shape of prf
			aperture = aperture.reshape(np.shape(prf))
		if plot:
			self._plot_aperture(prf_model, aperture, idx)
			
		return aperture
	
	# QUESTION: Should I make prf_model and aperture attributes of self? 
	# It's kind of a weird workflow calling 
	# tess_prf = lk.prf.TessPRF..., 
	# tprf = tp.prf_model...
	# aperture = tp.get_simple_aperture...
	# tp.estimate_conamination(tprf, aperture...
	
	# Should this be a hidden function?
	def estimate_contamination(self, prf_model:npt.ArrayLike, aperture: npt.ArrayLike, fluxes:npt.ArrayLike, idx:int=0) -> float:
		"""
		
		Parameters: 
		-----------
		prf_model : npt.ArrayLike
			3D cube of the PRFs of all sources with shape (nsources x nrows x ncolumns)
		aperture: npt.ArrayLike 
			2D boolean array same size as `prf`, True where source is inside aperture
		fluxes : npt.ArrayLike
			Array of fluxes for each source with shape (nsources)
		idx : int
			The index of the source to be considered as the target (default 0)

		Returns:
		--------
		contamination : float
			Fraction of total flux in the aperture that comes from the target source.
			Will be a value between 0 and 1 (1 being all flux comes from the target source)
		"""
		
		if prf_model.shape[0] == 1:
			print('There are no other sources in your TPF. Contamination will be 1 (no contamination)')
			return 1.
		
		prfs = np.array( [prf_model[ii, :, :] * fluxes[ii] for ii in range(len(fluxes))])
		
		target_flux = np.sum(prfs[idx, aperture])
		all_flux = np.sum(prfs[:, aperture] )
		return target_flux/all_flux
		
		
	def estimate_completeness(self, prf_model: npt.ArrayLike, aperture: npt.ArrayLike, idx: int = 0) -> float: 
		'''
		Returns the fraction of total flux from the target contained in a given aperture
		
		Parameters:
		-----------
		prf_model : npt.ArrayLike
			3D cube of the PRFs of all sources with shape (nsources x nrows x ncolumns)
		aperture: npt.ArrayLike 
			2D boolean array same size as the prf model, True where source is inside aperture
		idx : int
			The index of the source to be considered as the target (default 0)
			
		Returns:
		--------
		completeness : float
			fraction of total flux contained within the aperture
		'''
		return np.sum(prf_model[idx, aperture]) / np.sum(prf_model[idx,:,:])
		
		
		
	def evaluate(self,
		center_col = None, 
		center_row = None, 
		flux: float = 1.0,
		scale: float = 1.0,
		rotation_angle: float = 0.0):
		
		"""
		Interpolates the PRF model onto detector coordinates.

		Parameters
		----------
		center_col, center_row : float
			Column and row coordinates of the center
		flux : float
			Total integrated flux of the PRF
		scale : float
			Pixel scale stretch parameter, i.e. the numbers by which the PRF
			model needs to be multiplied in the column and row directions to
			account for focus changes
		rotation_angle : float
			Rotation angle in radians

		Returns
		-------
		prf : 2D array
			Two dimensional array representing the PRF values parametrized
			by flux, centroids, widths, and rotation as applicble.
		"""
		if center_col is None:
			center_col = self.column + self.shape[1] / 2
		if center_row is None:
			center_row = self.row + self.shape[0] / 2

		print(self.col_coord)
		delta_col = self.col_coord - center_col
		delta_row = self.row_coord - center_row
		print(delta_col)
		
		
		if (scale == 1.0) and (rotation_angle == 0.0):
			prf = flux * self.interpolate(delta_row, delta_col)		

		else:
			# Does this need to be a meshgrid if scale is required to be symmetrical?
			#rot_col, rot_row = np.meshgrid(delta_col, delta_row)
			rot_col = delta_col
			rot_row = delta_row.copy()
			if rotation_angle != 0:
				cosa = math.cos(rotation_angle)
				sina = math.sin(rotation_angle)
				rot_row = delta_row * cosa - rot_col * sina
				rot_col = delta_row * sina + rot_col * cosa
			
			# Why does this make the sum of the PRF << 1?
			# Changing this to divide by scale does more what I would expect
			# print(np.array(rot_row.flatten()) / scale)
			'''print(np.array(rot_col.flatten()) * 2)
			prf = flux * self.interpolate(
				np.array(rot_row.flatten()) * scale , np.array(rot_col.flatten()) * scale , grid=False
			).reshape(self.shape)'''
			prf = flux * self.interpolate(delta_row * scale, delta_col* scale)

		
		# NS maybe you could just augment the files so there is a border of 0s around
		# The prf values around the edges tend to be crazy small, so this may be more practical
		prf[prf < 1e-16] = 0.0 #Discuss this cutoff
		return prf
		
	def gradient(self,
		center_col = None, 
		center_row = None, 
		flux: float = 1.0,
		scale: float = 1.0,
		rotation_angle: float = 0.0) -> list:
		
		"""
		This function returns the gradient of the PRF model with
		respect to center_col, center_row, flux, scale,
		and rotation_angle.

		Parameters
		----------
		center_col, center_row : float
			Column and row coordinates of the center
		flux : float
			Total integrated flux of the PRF
		scale : float
			Pixel scale stretch parameter, i.e. the numbers by which the PRF
			model needs to be multiplied in the column and row directions to
			account for focus changes
		rotation_angle : float
			Rotation angle in radians

		Returns
		-------
		grad_prf : list
			Returns a list of arrays where the elements are the partial derivatives
			of the PRF model with respect to center_col, center_row, flux, scale_col,
			scale_row, and rotation_angle, respectively.
		"""
		if center_col is None:
			center_col = self.column + self.shape[1] / 2
		if center_row is None:
			center_row = self.row + self.shape[0] / 2

		
		delta_col = self.col_coord - center_col
		delta_row = self.row_coord - center_row
		
		
		
		if (scale == 1.0) and (rotation_angle == 0.0):
			deriv_flux = self.interpolate(delta_row, delta_col)
			deriv_center_col = -flux * self.interpolate(delta_row, delta_col, dy=1)
			deriv_center_row = -flux * self.interpolate(delta_row, delta_col, dx=1)

			return [deriv_center_col, deriv_center_row, deriv_flux]	

		else:
			# Does this need to be a meshgrid if scale is required to be symmetrical?
			delta_col, delta_row = np.meshgrid(delta_col, delta_row)
			rot_row = delta_row * cosa - delta_col * sina
			rot_col = delta_row * sina + delta_col * cosa
			
			# for a proof of the maths that follow, see the pdf attached
			# on pull request #198 in lightkurve GitHub repo.
			deriv_flux = self.interpolate(
				rot_row.flatten() * scale_row, rot_col.flatten() * scale_col, grid=False
			).reshape(self.shape)

			interp_dy = self.interpolate(
				rot_row.flatten() * scale_row,
				rot_col.flatten() * scale_col,
				grid=False,
				dy=1,
			).reshape(self.shape)

			interp_dx = self.interpolate(
				rot_row.flatten() * scale_row,
				rot_col.flatten() * scale_col,
				grid=False,
				dx=1,
			).reshape(self.shape)

			scale_row_times_interp_dx = scale_row * interp_dx
			scale_col_times_interp_dy = scale_col * interp_dy

			deriv_center_col = -flux * (
				cosa * scale_col_times_interp_dy - sina * scale_row_times_interp_dx
			)
			deriv_center_row = -flux * (
				sina * scale_col_times_interp_dy + cosa * scale_row_times_interp_dx
			)
			deriv_scale_row = flux * interp_dx * rot_row
			deriv_scale_col = flux * interp_dy * rot_col
			deriv_rotation_angle = flux * (
				interp_dy * scale_col * (delta_row * cosa - delta_col * sina)
				- interp_dx * scale_row * (delta_row * sina + delta_col * cosa)
			)

			return [
				deriv_center_col,
		   	 	deriv_center_row,
				deriv_flux,
				deriv_scale_col,
				deriv_scale_row,
				deriv_rotation_angle,
			]
		
	def prf_model(self, 
		center_col:  Union[float, list[float], npt.ArrayLike, None] = None, 
		center_row:  Union[float, list[float], npt.ArrayLike, None] = None,
		scale: float = 1.0,
		rotation_angle: float = 0.0 )-> npt.ArrayLike:
		'''
		Creates a cube of PRF models 
		
		Optional Parameters
		-------------------
		center_col : float or list of floats
			column location of the target on the CCD. If not provided, finds the center of the field of view
		center_row : float or list of floats
			row location of the target on the CCD. If not provided, finds the center of the field of view
		scale : float
			Pixel scale stretch parameter, i.e. the numbers by which the PRF
			model needs to be multiplied in the column and row directions to
			account for focus changes. Default is 1 (no scaling)
		rotation_angle : float
			Rotation angle in radians. default 0.0, ie no rotation
			
		Returns
		-------
		PRF model : npt.ArrayLike
			3D cube of PRF models of shape (ntargets, nrows, ncolumns) at instrument pixel resolution
		
		'''
		
		# PRF.evaluate returns 1 PRF. 
		# Here, evaluate is called for each target provided (e.g., for each source within a tpf) 

		if isinstance(center_col, (list, np.ndarray)):
			if len(center_col) != len(center_row):
				raise ValueError( "Column/row locations must have the same shape.")		
			prf_model = np.zeros((len(center_col), self.shape[0], self.shape[1])) 
			for ii in range(len(center_col)):
				# Flux for each target is not necessarily required
				prf_model[ii,:,:] = self.evaluate(center_col=center_col[ii], center_row=center_row[ii],  scale=scale, rotation_angle=rotation_angle)
			return prf_model
		else:
			prf_model = self.evaluate(center_col, center_row, scale=scale, rotation_angle=rotation_angle)
			return np.expand_dims(prf_model, axis=0)
			
	
	def _plot_aperture(self, prf_model: npt.ArrayLike, aperture: npt.ArrayLike, idx: int = 0, hatch_color: str = 'red'):
		fig, ax = plt.subplots(1)
		ax.set_title("Model PRF Aperture")
		ax.imshow(prf_model[idx, :,:], 
			origin='lower',
			extent=(
				self.column,
				self.column + self.shape[1],
				self.row,
				self.row + self.shape[0],
			)
		)
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				if aperture[i, j]:
					xy = (j + self.column, i + self.row)
					rect = patches.Rectangle(
						xy=xy, 
						width=1, 
						height=1, 
						color=hatch_color, 
						fill=False,
						hatch="//")
					ax.add_patch(rect)
		plt.show()
		
		
	
		
	@abstractmethod
	def _prepare_prf(self):
		"""Method to open PRF files for given mission"""
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
	The measured PRF is over-sampled by a factor of 50 to enable for sub-pixel interpolation.
	The model is a 550x550 (or 750x750) grid that covers 11x11 (or 15x15) pixels
	"""
	# I want the option to either give it a tpf and it reads channel/shape OR provide that info
	def __init__(self, column: int, row: int, channel:int, shape:tuple):
		super().__init__(column=column, row=row, shape=shape)
		self.channel = channel
		
		(
			self.col_coord,
			self.row_coord,
			self.interpolate,
			self.supersampled_prf,
		) = self._prepare_prf()
		
	def __repr__(self):
		return "I'm a Kepler PRF"
		
	def __call__(self, 
			center_col:  Union[float, list[float], npt.ArrayLike, None] = None, 
			center_row:  Union[float, list[float], npt.ArrayLike, None] = None,
			scale: float = 1.0,
			rotation_angle: float = 0.0 )-> npt.ArrayLike: # Add more here
		return self.prf_model(
			center_col, center_row, scale=scale, rotation_angle=rotation_angle
		)
		
		
	def _read_prf_calibration_file(self, path, ext):
		''' Reads the Kepler calibration files'''
		prf_cal_file = pyfits.open(path)
		data = prf_cal_file[ext].data
		crval1p = prf_cal_file[ext].header["CRVAL1P"]
		crval2p = prf_cal_file[ext].header["CRVAL2P"]
		cdelt1p = prf_cal_file[ext].header["CDELT1P"]
		cdelt2p = prf_cal_file[ext].header["CDELT2P"]
		prf_cal_file.close()

		return data, crval1p, crval2p, cdelt1p, cdelt2p
		
	def _prepare_prf(self):
		'''
		Sets up the PRF model interpolation by reading in the relevant Kepler files,
		and combining them by weighting them by distance to the location on the CCD of interest
		'''
		n_hdu = 5 # measurements at the 4 corners + center
		min_prf_weight = 1e-6
		module, output = channel_to_module_output(self.channel)
		# determine suitable PRF calibration file
		module = str(module).zfill(2)
		prfs_url_path = "../prf/data/kepler/kplr"
		
		prffile = (
			prfs_url_path
			+ str(module)
			+ "."
			+ str(output)
			+ "_2011265_prf.fits"
		)
		
		prffile = get_pkg_data_filename(prffile)
		
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
	
	@classmethod
	def from_tpf(self):
		'''Creates a PRF object to reflect the properties of a given TPF'''
		# Add some error checks that it's a Kepler TPF here
		prf_object =  KeplerPRF(self.column, self.row, self.channel, self.shape[1:3])
		return prf_object
		
	def plot(self, *params): 
		""" Plots the supersampled PRF model, evaluated for the given location on the CCD

		Parameters:
		-----------
		Can use optional formatting parameters used by lk.utils.plot_image

		Returns:
		--------
		ax : `~matplotlib.axes.Axes`
			The matplotlib axes object.
		"""
		
		ax = plot_image(
			self.supersampled_prf,
			title=f"Supersampled Kepler PRF Model, Channel: {self.channel}",
			extent=(
				self.column,
				self.column + self.shape[1],
				self.row,
				self.row + self.shape[0],
			),
			clabel='relative flux',
			*params
		)	
		return ax

	

		
#############################################################		
#############################################################
#						 TESS PRF
#############################################################
#############################################################
class TessPRF(PRF):
	"""A TessPRF class""" 
	def __init__(self, 
				column: int, 
				row: int, 
				shape:tuple,
				camera: int, 
				ccd: int
				):
		# Sector needed as different engineering files are needed for sectors < 4
		super().__init__(column=column, row=row, shape=shape)
		self.camera = camera
		self.ccd = ccd
		
		(
			self.col_coord,
			self.row_coord,
			self.interpolate,
			self.supersampled_prf,
		) = self._prepare_prf()
		
		
	def __repr__(self):
		return "I'm a TESS PRF"
		
	def __call__(self, 
			center_col:  Union[float, list[float], npt.ArrayLike, None] = None, 
			center_row:  Union[float, list[float], npt.ArrayLike, None] = None,
			scale: float = 1.0,
			rotation_angle: float = 0.0 )-> npt.ArrayLike: # Add more here
		return self.prf_model(
			center_col, center_row, scale=scale, rotation_angle=rotation_angle
		)
	
	def _read_prf_calibration_file(self, hdu):
		
		data = hdu.data
		crval1p = hdu.header["CRVAL1P"]
		crval2p = hdu.header["CRVAL2P"]
		cdelt1p = hdu.header["CDELT1P"]
		cdelt2p = hdu.header["CDELT2P"]
		
		return data, crval1p, crval2p, cdelt1p, cdelt2p	
		
	def _prepare_prf(self):
		'''
		Sets up the PRF model interpolation by reading in the relevant TESS files,
		and combining them by weighting them by distance to the location on the 
		camera/CCD of interest
		'''
		min_prf_weight = 1e-6 


		# NOTE: now treating all sectors the same (always use the > sector 4 fils)
		# This file has the prf images saved in extensions 1-25 (not 0-24)
		prffile = f"../prf/data/tess/tess_prf_cam{self.camera}_ccd{self.ccd}.fits"
		prffile = get_pkg_data_filename(prffile)
		prf_cal_file = pyfits.open(prffile)
		
		# TESS has a separate file for each point in a grid of pixel locations.
		# Find the closest 4 to your pixel location and go from there.
		rows = np.array([prf_cal_file[i].header['CRVAL2P'] for i in range(1, 26)])
		cols = np.array([prf_cal_file[i].header['CRVAL1P'] for i in range(1, 26)])
		
		distance = np.sqrt((rows-self.row)**2. + (cols-self.column)**2.)
		# Provide the index for the row/column combination that make up a box around the target location
		nearest_four = np.argpartition(distance,4)[:4] 

		n_hdu = 4
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
			) = self._read_prf_calibration_file(prf_cal_file[nearest_four[i]+1])
		prf_cal_file.close()
		
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

		interpolate = scipy.interpolate.RectBivariateSpline(PRFrow, PRFcol, supersamp_prf)
			
		 
		return col_coord, row_coord, interpolate, supersamp_prf

	def from_tpf(self):
		'''Creates a PRF object using a TPF'''
		# Add some error checks that it's a Tess TPF here?
		return TessPRF(self.column, self.row, self.camera, self.ccd)
		
	def plot(self, *params): # Fill in params explicitly
		""" Plots the supersampled PRF model, evaluated for the given location on the CCD

		Parameters:
		-----------
		Can use optional formatting parameters used by lk.utils.plot_image

		Returns:
		--------
		ax : `~matplotlib.axes.Axes`
			The matplotlib axes object.
		"""
		ax = plot_image(
			self.supersampled_prf,
			title=f"TESS PRF Model, Camera: {self.camera}, CCD: {self.ccd}",
			extent=(
				self.column,
				self.column + self.shape[1],
				self.row,
				self.row + self.shape[0],
			),
			clabel='log relative flux',
			*params
		)
		
		return ax

		
		
