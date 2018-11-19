"""Tools to interact more easily with Full Frame Image files."""
from __future__ import division, print_function

from astropy.io import fits
from astropy.wcs import WCS

__all__ = ['FullFrameImage']


def FullFrameImage(object):

    def __init__(self, path_or_url, **kwargs):
        self.path = path_or_url
        if isinstance(path_or_url, fits.HDUList):
            self.hdulist = path_or_url
        else:
            self.hdulist = fits.open(self.path, **kwargs)

    def skycoord_to_pixel(self, ra, dec):
        """Converts ra, dec to channel, x, y"""
        for channel in range(1, 85):
            hdr = self.hdulist[channel].header
            wcs = WCS(hdr)
            (col, row) = wcs.all_world2pix([ra], [dec], 0)
            if (col > 0) and (col < hdr['NAXIS1']) and (row > 0) and (row < hdr['NAXIS2']):
                return channel, col, row
        return None, None, None

    def plot_cutout(self, coord, size=10):
        pass

    def plot():
        """Show all channels."""
        pass
