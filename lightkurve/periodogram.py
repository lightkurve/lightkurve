import numpy as np
from astropy import units as u
from astropy.stats import LombScargle
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt

__all__ = ['Periodogram']

class Periodogram(object):
	"""
    Implements a simple class for a Periodogram

    Attributes
    ----------
    lightcurve : object containing flux, time, etc.
        lightcurve object
    frequency: array-like
        frequency for every time point
    model : astropy obj
        A model of the power that can be generated
    power : astropy Units array-like
        Power measurements
    """
	def __init__(self, frequency=None, model=LombScargle, 
				power=None, lc=None):
		self.lightcurve = lc
		nyquist_frequency = 0.5 * (1./((np.median(self.lightcurve.time[1:] - self.lightcurve.time[0:-1])*u.day).to(u.second))).to(u.microhertz).value
		frequency = np.linspace(1, nyquist_frequency, len(self.lightcurve.time)//2) * u.microhertz
		self.frequency = frequency
		try:
			self.model = model((self.lightcurve.time*u.day).to(u.second), self.lightcurve.flux*1e6)
		except:
			raise AttributeError("No Lightcurve given")
		self.power = power

	def from_lightcurve(lc):
		"""Creates a Periodogram object from a lightcurve"""
		return Periodogram(lc=lc)
	
	def generate_power(self, frequency):
		"""Uses Lomb-Scargle (by default) to generate the power spectrum for a given frequency

		Parameters
		----------
		frequency: Astropy Units, array-like
			Frequency in microhertz over the range you want to assess

		Sets the power spectrum using Lomb Scargle
		"""
		uHz_conv = 1./(((1./u.day).to(u.microhertz)))
		self.power = self.model.power(frequency, method="fast", normalization="psd")
		self.power *= uHz_conv / len(self.lightcurve.time)  # Convert to ppm^2/uHz

	def generate_freq(self, frequency):
		"""
		Generates a frequency if none is provided.
		Else this method will convert the frequency given into microhertz.
		This method will attempt to convert time series in days to a frequency in microhertz.
		
		Parameters
		----------
		frequency: array-like
			Frequency in astropy.units.microhertz

		Returns
		-------
		frequency: astropy.units.microhertz array-like
			Frequency with correct microhertz conversions up until the nyquist frequency if unspecified range

		"""
		if frequency is None:
			nyquist_frequency = 0.5 * (1./((np.median(self.lightcurve.time[1:] - self.lightcurve.time[0:-1])*u.day).to(u.second))).to(u.microhertz).value
			frequency = np.linspace(1, nyquist_frequency, len(self.lightcurve.time)//2) * u.microhertz
		else:
			frequency = np.asarray(frequency) #Load as a numpy array
			if type(frequency) != u.quantity.Quantity: #Has no astropy units
				frequency = frequency * u.microhertz
			else:
				if frequency.unit != "uHz": #try to convert from DAYS to MICROHERTZ
					frequency *= 1./u.day
					frequency *= 1./uHz_conv
		return frequency

	def plot(self, frequency=None, scale="linear", ax=None, numax=None, **kwargs):
		"""Plots the periodogram.

		Parameters
		----------
		frequency: array-like
			Over what frequencies (in microhertz) will periodogram plot
		scale: str
			Set x,y axis to be "linear" or "log". Default is linear.
		ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        numax: bool
        	Plot the numax value as well?
		kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

		Returns
		-------
		ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.

        """

		if ax is None:
			fig, ax = plt.subplots()
		#Generate frequency and associated power spectrums.
		frequency = self.generate_freq(frequency)
		self.generate_power(frequency)
		#Plot frequency and power
		ax.plot(frequency, self.power, **kwargs)
		ax.set_xlabel("Frequency [$\mu$Hz]")
		ax.set_ylabel("Power [ppm$^2$/$\mu$Hz]");
		#Attempt to create a title based on KIC
		try:
			ax.set_title('KIC {}'.format(self.lightcurve.keplerid))
		except AttributeError:
			print("No KIC was found")
		
		if numax:
			ax.fill_between([numax.value*0.8, numax.value*1.2], self.power.value.min(), self.power.value.max(), alpha=0.2, color='C3', zorder=10)
		if scale == "log":
			ax.set_yscale('log')
			ax.set_xscale('log')
		return ax