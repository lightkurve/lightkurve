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
I want to build a PRF model to simulate a real TPF
	- call PRF(tpf). This gets channel/cam/ccd, col/row from tpf header
	- use Rebekah's search_grid function (in development) to get a table of all stars
		brighter than a given magnitude cutoff
		- the returned table includes ra/dec with pm correction, pixel x and y, and TESS magnitude
	- loop through stars and create a prf for each star, with flux determined by the TESS mag
	- return a PRF that sums all of the PRFs
	
I want to just see what the PRF looks like at a given location
	- call PRF(channel/cam/cdd, row, col OR ra, dec)
	- return a PRF the size of the engineering files (11x11 pixels I believe)
	
I want to create a PRF of only a selection of stars
    - call PRF(table, channel/camera/ccd)
    	- Note table would be in the same format as Rebekah's return table
	- loop through stars and create a prf for each star
	- How would I know how big to make this? 
		- If not specified, I guess just use the largest/smallest columns and rows?



# New Workflow:
# Instantiate PRF with only the channel (Kepler) or camera/ccd (TESS)
# Call a 'model' function with the column, row, flux, center_col, center_row, flux, scale_col, scale_row, rotation_angle


# Just sketching out a few models right now
def class PRF(ABC):

	def estimate_aperture(tpf):
		# Given a tpf, build a prf model and estimate the best aperture
	def create_simple_aperture(pix_c, pix_r, ra, dec, flux, completeness)
		# Based on completeness requirement, create an aperture
		# This wouldn't worry about contamination I think
	def model(self,
		corner_col,
		corner_row,
		center_col, # if not provided, use corner_col + self.shape[0] / 2
        center_row, # if not provided, use corner_row + self.shape[1] / 2
        flux=1.0,
        scale_col=1.0,
        scale_row=1.0,
        rotation_angle=0.0,
    ):
		
'''

		
#############################################################        
#############################################################
#                        KEPLER PRF
#############################################################
#############################################################
class KeplerPRF(PRF):
    """A KeplerPRF class"""
    # I want the option to either give it a tpf and it reads channel/shape OR provide that info
    def __init__(self, channel, shape):
    	self.channel = channel
    	self.shape = shape
    	
    def __repr__(self):
        return "I'm a Kepler PRF"
        
    def __call__(self, center_col, center_row, **kwargs):
        return self.evaluate(
            center_col, center_row, **kwargs
        )

    	
#############################################################        
#############################################################
#                         TESS PRF
#############################################################
#############################################################
class TessPRF(PRF):
    """A TessPRF class""" 
    def __init__(self, camera, ccd, shape):
    	self.camera = camera
    	self.ccd = ccd
    	self.shape = shape
    	
    def __repr__(self):
        return "I'm a TESS PRF"
        
    def __call__(self, center_col, center_row, **kwargs):
        return self.evaluate(
            center_col, center_row, **kwargs
        )

    	


		
		





















################################################################################
#                                   KEPLER  
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
#                                   TESS  
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
