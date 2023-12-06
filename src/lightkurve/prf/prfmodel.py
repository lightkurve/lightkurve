""" 
This is an update to the PRF module. Changes include:

- Added functionality for TESS
- TESS PRF files are modified to '0-out' pixels that do not contribute to the PRF, matching Kepler's format
- Kepler and TESS files are stored in the lightkurve file structure (ie not downloaded from the internet)
- PRF contributions < 1e-16 are ignored.
- Users can use prf.from_tpf(TPF) to create a PRF object directly from Kepler/K2/TESS Target Pixel Files
- Additional functions allow users to 
	- calculate a simple aperture
	- estimate completeness
	- estimate contamination
- targetpixelfile.py adds functionality to create PRF models matching the TPF field of view


Use cases :

"I want to make an aperture for a TESSCut TPF"
	- tpf.get_target_aperture(...)
	
"I want to build a PRF model to simulate a real TPF"
	- If given a list of the location of each object within the cutout region, 
		prf_model returns a 3D cube contining PRF models for each object. 
	- get_flux_weighted_prf() multiplies the prf_model cube by the list of fluxes
	- This can then be summed to get the prf model of the field. 
	

"I want to just see what the PRF looks like at a given location"
	- lk.prf.TessPRF(column = 100, row = 200, camera = 1, ccd = 1).plot()
	- lk.prf.TessPRF(column = 100, row = 200, camera = 1, ccd = 1).prf_model(center_col=105,
		center_row=202)
	


"""
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
import warnings
from ..utils import LightkurveWarning


def from_tpf(tpf):
	"""
	Returns a KeplerPRF or TessPRF object when passed TPF

	Parameters:
	-----------
	tpf : TargetPixelFile
			lightkurve TargetPixelFile of a TESS, Kepler, or K2 cutout

	Returns:
	--------
	prf_object : PRF
			Initialized PRF object in the shape and location of the TPF
	"""

	if tpf.mission.lower() == "tess":
		prf_object = TessPRF(tpf.column, tpf.row, tpf.camera, tpf.ccd, tpf.shape[1:3])
	elif (tpf.mission.lower() == "kepler") | (tpf.mission.lower() == "k2"):
		prf_object = KeplerPRF(tpf.column, tpf.row, tpf.channel, tpf.shape[1:3])
	else:
		raise TypeError("TPF must be from Kepler/K2 or TESS")

	return prf_object


class PRF(ABC):
	@abstractmethod
	def __init__(self, column: int, row: int, shape: tuple = (11, 11)):
		"""
		A generic base class object for PRFs. No to be used directly.
		See KeplerPRF and TessPRF for the instantiable classes.

		Parameters:
		-----------
		column : int
			pixel coordinate of the lower left column value
		row : int
			pixel coordinate of the lower left row value
		shape : tuple
			shape of the resultant PRFs in pixels
		"""

		# load PSF file
		# Initialize object
		self.column = column
		self.row = row
		self.shape = shape

	def __repr__(self):
		return "PRF Base Class"

	def get_target_aperture(
		self,
		prf_model: npt.ArrayLike,
		target_idx: int = 0,
		min_completeness: float = 0.9,
	) -> npt.ArrayLike:
		"""
		Based on completeness requirement, create an aperture.
		This basic aperture does NOT account for contamination by other sources.

		Parameters:
		-----------
		prf_model : npt.ArrayLike
			3D cube of the PRFs of all sources with shape (nsources x nrows x ncolumns)
			Only the target source is used for this function.
		target_idx : int
			The index of the source to be considered as the target (default 0)
		min_completeness : float
			Minimum fraction of flux contained within the aperture. Value between 0 and 1


		Returns:
		--------
		aperture : npt.ArrayLike
			2D boolean array of the same (nrows x ncolumns) as `prf_model`.
			True where source is inside aperture
		"""
		
		if (min_completeness < 0) | (min_completeness > 1):
			raise ValueError("Completeness must be between 0 and 1")
			
			
		prf = prf_model[target_idx, :, :]


		if np.sum(prf) > 1.0:
			prf = prf / np.sum(prf)

		# If completeness = 1, return any pixel that contains any amount of flux
		if min_completeness == 1.0:
			aperture = prf.astype(bool)
			aperture[prf != 0.0] = True

		else:
			sort = np.argsort(prf.flatten())
			cusu = np.cumsum(prf.flatten()[sort])

			# indices that are "inside" aperture
			ap_index = sort[(1 - cusu) < min_completeness]
			aperture = np.zeros(prf.shape, dtype=bool).flatten()
			aperture[ap_index] = True
			# reshape to shape of prf
			aperture = aperture.reshape(np.shape(prf))
		if plot:
			self._plot_aperture(prf_model, aperture, target_idx)

		return aperture

	def get_contamination(
		self,
		prf_model: npt.ArrayLike,
		aperture: npt.ArrayLike,
		fluxes: npt.ArrayLike,
		target_idx: int = 0,
	) -> float:
		"""

		Parameters:
		-----------
		prf_model : npt.ArrayLike
						3D cube of the PRFs of all sources with shape (nsources x nrows x ncolumns)
		aperture: npt.ArrayLike
						2D boolean array same size as `prf`, True where source is inside aperture
		fluxes : npt.ArrayLike
						Array of fluxes for each source with shape (nsources)
		target_idx : int
						The index of the source to be considered as the target (default 0)

		Returns:
		--------
		contamination : float
						Fraction of total flux in the aperture that comes from the target source.
						Will be a value between 0 and 1 (1 being all flux comes from the target source)
		"""

		if prf_model.shape[0] == 1:
			print(
				"Only a target source is provided. Contamination will be 1 (no contamination)"
			)
			return 1.0

		prfs = self.get_flux_weighted_prf(prf_model, fluxes)

		target_flux = np.sum(prfs[target_idx, aperture])
		all_flux = np.sum(prfs[:, aperture])
		return target_flux / all_flux

	def get_flux_weighted_prf(self,
		prf_model: npt.ArrayLike, 
		fluxes: npt.ArrayLike
	) -> npt.ArrayLike:
		"""

		Parameters:
		-----------
		prf_model : npt.ArrayLike
						3D cube of the PRFs of all sources with shape (nsources x nrows x ncolumns)
		fluxes : npt.ArrayLike
						Array of fluxes for each source with shape (nsources)

		Returns:
		--------
		prf_with_flux : npt.ArrayLike
						prf model multiplied by total expected flux

		"""
		if len(fluxes) != prf_model.shape[0]:
			raise ValueError(
				f"number of input fluxes do not match number of elements in prf_model."
			)
		return np.array([prf_model[ii, :, :] * fluxes[ii] for ii in range(len(fluxes))])


	def get_completeness(
		self, prf_model: npt.ArrayLike, aperture: npt.ArrayLike, target_idx: int = 0
	) -> float:
		"""
		Returns the fraction of total flux from the target contained in a given aperture

		Parameters:
		-----------
		prf_model : npt.ArrayLike
						3D cube of the PRFs of all sources with shape (nsources x nrows x ncolumns)
		aperture: npt.ArrayLike
						2D boolean array same size as the prf model, True where source is inside aperture
		target_idx : int
						The index of the source to be considered as the target (default 0)

		Returns:
		--------
		completeness : float
						fraction of total flux contained within the aperture
		"""
		return np.sum(prf_model[target_idx, aperture]) / np.sum(
			prf_model[target_idx, :, :]
		)

	def evaluate(
		self,
		center_col=None,
		center_row=None,
		scale: float = 1.0,
		rotation_angle: float = 0.0,
	):
		"""
		Interpolates the PRF model onto detector coordinates.

		Parameters
		----------
		center_col, center_row : float
						Column and row coordinates of the center
		scale : float
						Pixel scale stretch parameter, can be used to account for focus changes.
						Values > 1 stretch the image, Values < 1 make the PRF more compact.
						E.g. a scale value of 2 will double the PRF footprint.
		rotation_angle : float
						Rotation angle in radians

		Returns
		-------
		prf : 2D array
						Two dimensional array representing the PRF values parametrized
						by flux, centroids, widths, and rotation as applicble.
		"""
		if scale <= 0:
			scale = 1
			warnings.warn(
				f"Scale can not be <= 0. Resetting scale to 1.",
				LightkurveWarning,
			)
			
		scale = 1.0 / scale

		if center_col is None:
			center_col = self.column + self.shape[1] / 2
		if center_row is None:
			center_row = self.row + self.shape[0] / 2

		delta_col = self.col_coord - center_col
		delta_row = self.row_coord - center_row

		if (scale == 1.0) and (rotation_angle == 0.0):
			prf = self.interpolate(delta_row, delta_col)

		else:
			cosa = math.cos(rotation_angle)
			sina = math.sin(rotation_angle)

			delta_col, delta_row = np.meshgrid(delta_col, delta_row)
			rot_row = delta_row * cosa - delta_col * sina
			rot_col = delta_row * sina + delta_col * cosa

			prf = self.interpolate(
				rot_row.flatten() * scale, rot_col.flatten() * scale, grid=False
			).reshape(self.shape)
			
			prf = prf / (1 / scale) ** 2
			
			# Normalize the values when 'scale' is set to decrease the PRF spread
			if (1 / scale) < 1:
				prf = prf/np.sum(prf)
			

		# Ignore relative flux below a given threshold as the resulting flux change is not detectable
		prf[prf < 1e-16] = 0.0
		return prf

	def gradient(
		self,
		center_col=None,
		center_row=None,
		flux: float = 1.0,
		scale: float = 1.0,
		rotation_angle: float = 0.0,
	) -> list:
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
						account for focus changes.
						Values > 1 stretch the image, Values < 1 make the PRF more compact.
						E.g. a scale value of 2 will double the PRF footprint.
		rotation_angle : float
						Rotation angle in radians

		Returns
		-------
		grad_prf : list
						Returns a list of arrays where the elements are the partial derivatives
						of the PRF model with respect to center_col, center_row, flux, scale_col,
					scale_row, and rotation_angle, respectively.
		"""
		
		if scale <= 0:
			scale = 1
			warnings.warn(
				f"Scale can not be <= 0. Resetting scale to 1.",
				LightkurveWarning,
			)
		# Implemented to match intuition that larger scale value results in a broader PRF
		scale = 1.0 / scale

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
			delta_col, delta_row = np.meshgrid(delta_col, delta_row)
			rot_row = delta_row * cosa - delta_col * sina
			rot_col = delta_row * sina + delta_col * cosa

			# for a proof of the maths that follow, see the pdf attached
			# to pull request #198 in lightkurve GitHub repo.
			deriv_flux = self.interpolate(
				rot_row.flatten() * scale, rot_col.flatten() * scale, grid=False
			).reshape(self.shape)

			interp_dy = self.interpolate(
				rot_row.flatten() * scale,
				rot_col.flatten() * scale,
				grid=False,
				dy=1,
			).reshape(self.shape)

			interp_dx = self.interpolate(
				rot_row.flatten() * scale,
				rot_col.flatten() * scale,
				grid=False,
				dx=1,
			).reshape(self.shape)

			scale_row_times_interp_dx = scale * interp_dx
			scale_col_times_interp_dy = scale * interp_dy

			deriv_center_col = -flux * (
				cosa * scale_col_times_interp_dy - sina * scale_row_times_interp_dx
			)
			deriv_center_row = -flux * (
				sina * scale_col_times_interp_dy + cosa * scale_row_times_interp_dx
			)
			deriv_scale_row = flux * interp_dx * rot_row
			deriv_scale_col = flux * interp_dy * rot_col
			deriv_rotation_angle = flux * (
				interp_dy * scale * (delta_row * cosa - delta_col * sina)
				- interp_dx * scale * (delta_row * sina + delta_col * cosa)
			)

			return [
				deriv_center_col,
				deriv_center_row,
				deriv_flux,
				deriv_scale_col,
				deriv_scale_row,
				deriv_rotation_angle,
			]

	def _prf_model(
		self,
		center_col: Union[float, list[float], npt.ArrayLike],
		center_row: Union[float, list[float], npt.ArrayLike],
		scale: float = 1.0,
		rotation_angle: float = 0.0,
	) -> npt.ArrayLike:
		"""
		Creates a stack of PRF models.
		if center_col/center_row are lists (e.g., a list of pixel locations for each star
		located in a TPF), a prf will be generated for each.

		Optional Parameters
		-------------------
		center_col : float or list of floats
						column location of the target on the CCD. 
		center_row : float or list of floats
						row location of the target on the CCD. 
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

		"""

		# PRF.evaluate returns a PRF for one onject.
		# Here, evaluate is called for each target provided (e.g., for each source within a tpf)

		if isinstance(center_col, (list, np.ndarray)):
			if len(center_col) != len(center_row):
				raise ValueError("Column/row locations must have the same shape.")
			prf_model = np.zeros((len(center_col), self.shape[0], self.shape[1]))
			for ii in range(len(center_col)):
				# Flux for each target is NOT taken into account by default.
				# To account for flux, see get_flux_weighted_prf().
				prf_model[ii, :, :] = self.evaluate(
					center_col=center_col[ii],
					center_row=center_row[ii],
					scale=scale,
					rotation_angle=rotation_angle,
				)
			return prf_model
		else:
			prf_model = self.evaluate(
				center_col, center_row, scale=scale, rotation_angle=rotation_angle
			)
			return np.expand_dims(prf_model, axis=0)


	@abstractmethod
	def _prepare_prf(self):
		"""Method to open PRF files for given mission"""
		pass

	@abstractmethod
	def _read_prf_calibration_file(self):
		"""Method to read PRF fits files for each mission and extract needed information"""
		pass


#############################################################
#############################################################
# 						KEPLER PRF
#############################################################
#############################################################
class KeplerPRF(PRF):
	"""
	A KeplerPRF class. Can be used for Kepler or K2.

	There are 5 PRF measurements (the 4 corners and the center) for each channel.
	The measured PRF is over-sampled by a factor of 50 to enable for sub-pixel interpolation.
	The model is a 550x550 (or 750x750) grid that covers 11x11 (or 15x15) pixels

	https://archive.stsci.edu/missions/kepler/commissioning_prfs/
	"""

	def __init__(self, column: int, row: int, channel: int, shape: tuple = (11, 11)):
		super().__init__(column=column, row=row, shape=shape)
		self.channel = channel

		(
			self.col_coord,
			self.row_coord,
			self.interpolate,
			self.supersampled_prf,
		) = self._prepare_prf()
		

	def __repr__(self):
		return f"KeplerPRF Object from channel {self.channel} at column {self.column}, row {self.row}"

	def __call__(
		self,
		center_col: Union[float, list[float], npt.ArrayLike, None] = None,
		center_row: Union[float, list[float], npt.ArrayLike, None] = None,
		scale: float = 1.0,
		rotation_angle: float = 0.0,
	) -> npt.ArrayLike:  
		return self.prf_model(
			center_col, center_row, scale=scale, rotation_angle=rotation_angle
		)
		
	def prf_model(self,
		center_col: Union[float, list[float], npt.ArrayLike, None] = None,
		center_row: Union[float, list[float], npt.ArrayLike, None] = None,
		scale: float = 1.0,
		rotation_angle: float = 0.0,
		) -> npt.ArrayLike:
		PRF_mod = self._prf_model(center_col, center_row, scale=scale, rotation_angle=rotation_angle)
		
		# Mask out flux if it falls on the collateral pixels
		# See https://nexsci.caltech.edu/workshop/2012/keplergo/PipelineCAL.shtml
		tar_flux_init = np.sum(PRF_mod[0,:,:])
		if self.column < 12:
			mask_cols = 12 - self.column
			PRF_mod[:,:,:mask_cols] = 0
			warnings.warn(
				f"TPF contains collateral pixels: Column(s) < 12",
				LightkurveWarning,
			)
		if self.column + self.shape[1] >= 1112:
			mask_cols = 1112 - self.column
			PRF_mod[:,:,mask_cols:] = 0
			warnings.warn(
				f"TPF contains collateral pixels: Column(s) >= 1112 ",
				LightkurveWarning,
			)
		if self.row < 20:
			mask_rows = 20 - self.row
			PRF_mod[:,:mask_rows,:] = 0
			warnings.warn(
				f"TPF contains collateral pixels: Row(s) < 20",
				LightkurveWarning,
			)
		if self.row + self.shape[1] >= 1044:
			mask_rows = 1044 - self.row
			PRF_mod[:,mask_rows:,:] = 0
			warnings.warn(
				f"TPF contains collateral pixels: Row(s) >= 1044)",
				LightkurveWarning,
			)	  
		tar_flux_after = np.sum(PRF_mod[0,:,:])  
	  	
		if tar_flux_init != tar_flux_after:
			print(f"{100 * (1 - (tar_flux_after / tar_flux_init))}% of target flux fell on non-science pixels.")	  	
		return PRF_mod

	def _read_prf_calibration_file(self, path, ext):
		"""Reads the Kepler calibration files"""
		prf_cal_file = pyfits.open(path)
		data = prf_cal_file[ext].data
		crval1p = prf_cal_file[ext].header["CRVAL1P"]
		crval2p = prf_cal_file[ext].header["CRVAL2P"]
		cdelt1p = prf_cal_file[ext].header["CDELT1P"]
		cdelt2p = prf_cal_file[ext].header["CDELT2P"]
		prf_cal_file.close()

		return data, crval1p, crval2p, cdelt1p, cdelt2p

	def _prepare_prf(self):
		"""
		Sets up the PRF model interpolation by reading in the relevant Kepler files,
		and combining them by weighting them by distance to the location on the CCD of interest
		"""
		n_hdu = 5  # measurements at the 4 corners + center
		min_prf_weight = 1e-6
		module, output = channel_to_module_output(self.channel)
		# determine suitable PRF calibration file
		module = str(module).zfill(2)
		prfs_url_path = "../prf/data/kepler/kplr"

		prffile = prfs_url_path + str(module) + "." + str(output) + "_2011265_prf.fits"

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
		interpolate = scipy.interpolate.RectBivariateSpline(
			PRFrow, PRFcol, supersamp_prf
		)

		return col_coord, row_coord, interpolate, supersamp_prf

	def plot(self, *params):
		"""Generates a plot showing the supersampled PRF model, evaluated for the given location on the CCD

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
				0,
				self.supersampled_prf.shape[1]/50,
				0,
				self.supersampled_prf.shape[0]/50,
			),
			clabel="Relative Flux",
			*params,
		)
		return ax


#############################################################
#############################################################
# 						 TESS PRF
#############################################################
#############################################################
class TessPRF(PRF):
	"""A TessPRF class. The TESS PRF measurements are supersampled by a factor of 9."""

	def __init__(
		self, column: int, row: int, camera: int, ccd: int, shape: tuple = (11, 11)
	):
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
		return f"TessPRF Object from camera {self.camera}, ccd {self.ccd} at column {self.column}, row {self.row}"

	def __call__(
		self,
		center_col: Union[float, list[float], npt.ArrayLike, None] = None,
		center_row: Union[float, list[float], npt.ArrayLike, None] = None,
		scale: float = 1.0,
		rotation_angle: float = 0.0,
	) -> npt.ArrayLike:  # Add more here
		return self.prf_model(
			center_col, center_row, scale=scale, rotation_angle=rotation_angle
		)
		
	def prf_model(self,
		center_col: Union[float, list[float], npt.ArrayLike, None] = None,
		center_row: Union[float, list[float], npt.ArrayLike, None] = None,
		scale: float = 1.0,
		rotation_angle: float = 0.0,
		) -> npt.ArrayLike:
		PRF_mod = self._prf_model(center_col, center_row, scale=scale, rotation_angle=rotation_angle)
		
		# Mask out flux if it falls on the collateral pixels
		# See https://heasarc.gsfc.nasa.gov/docs/tess/data-products.html
		tar_flux_init = np.sum(PRF_mod[0,:,:])
		
		if self.column < 45:
			mask_cols = 45 - self.column
			PRF_mod[:,:,:mask_cols] = 0
			warnings.warn(
				f"TPF contains collateral pixels: Column(s) < 45",
				LightkurveWarning,
			)
		if self.column + self.shape[1] >= 2093:
			mask_cols = 2093 - self.column
			PRF_mod[:,:,mask_cols:] = 0
			warnings.warn(
				f"TPF contains collateral pixels: Column(s) >= 2093 ",
				LightkurveWarning,
			)
	  	  	  
		if self.row + self.shape[1] > 2048:
			mask_rows = 2048 - self.row
			PRF_mod[:,mask_rows:,:] = 0
			warnings.warn(
				f"TPF contains collateral pixels: Row(s) > 2048",
				LightkurveWarning,
			)
		tar_flux_after = np.sum(PRF_mod[0,:,:])
	  	  
		if tar_flux_init != tar_flux_after:
			print(f"{100 * (1 - (tar_flux_after / tar_flux_init))}% of target flux fell on non-science pixels.")	
		return PRF_mod 

	def _read_prf_calibration_file(self, hdu):
		data = hdu.data
		crval1p = hdu.header["CRVAL1P"]
		crval2p = hdu.header["CRVAL2P"]
		cdelt1p = hdu.header["CDELT1P"]
		cdelt2p = hdu.header["CDELT2P"]

		return data, crval1p, crval2p, cdelt1p, cdelt2p

	def _prepare_prf(self):
		"""
		Sets up the PRF model interpolation by reading in the relevant TESS files,
		and combining them by weighting them by distance to the location on the
		camera/CCD of interest
		"""
		min_prf_weight = 1e-6

		# NOTE: now treating all sectors the same (always use the > sector 4 fils)
		# This file has the prf images saved in extensions 1-25 (not 0-24)
		prffile = f"../prf/data/tess/tess_prf_cam{self.camera}_ccd{self.ccd}.fits"
		prffile = get_pkg_data_filename(prffile)
		prf_cal_file = pyfits.open(prffile)

		# TESS has a separate file for each point in a grid of pixel locations.
		# Find the closest 4 to your pixel location and go from there.
		rows = np.array([prf_cal_file[i].header["CRVAL2P"] for i in range(1, 26)])
		cols = np.array([prf_cal_file[i].header["CRVAL1P"] for i in range(1, 26)])

		distance = np.sqrt((rows - self.row) ** 2.0 + (cols - self.column) ** 2.0)
		# Provide the index for the row/column combination that make up a box around the target location
		nearest_four = np.argpartition(distance, 4)[:4]
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
			) = self._read_prf_calibration_file(prf_cal_file[nearest_four[i] + 1])
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

		supersamp_prf /= np.nansum(supersamp_prf) * cdelt1p[0] * cdelt2p[0]

		# location of the data image centered on the PRF image (in PRF pixel units)
		col_coord = np.arange(self.column + 0.5, self.column + coldim + 0.5)
		row_coord = np.arange(self.row + 0.5, self.row + rowdim + 0.5)
		# x-axis correspond to row-axis in scipy.RectBivariate
		# not to be confused with our convention, in which the
		# x-axis correspond to the column-axis

		interpolate = scipy.interpolate.RectBivariateSpline(
			PRFrow, PRFcol, supersamp_prf
		)

		return col_coord, row_coord, interpolate, supersamp_prf

	def plot(self, *params):  # Fill in params explicitly
		"""Plots the supersampled PRF model, evaluated for the given location on the CCD

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
				0,
				self.supersampled_prf.shape[1]/9,
				0,
				self.supersampled_prf.shape[0]/9,
			),
			clabel="Relative Flux",
			*params,
		)

		return ax
