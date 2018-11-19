"""Tools to interact more easily with Full Frame Image files."""
from __future__ import division, print_function

from astropy.io import fits
from astropy.wcs import WCS

from .utils import plot_image

__all__ = ['FullFrameImage']


class FullFrameImage(object):

    def __init__(self, path_or_url, **kwargs):
        self.path = path_or_url
        if isinstance(path_or_url, fits.HDUList):
            self.hdulist = path_or_url
        else:
            self.hdulist = fits.open(self.path, **kwargs)

    def skycoord_to_pixel(self, skycoord):
        """Converts an (ra, dec) sky coordinate to a (channel, column, row) tuple."""
        for channel in range(1, 85):
            hdr = self.hdulist[channel].header
            if 'CTYPE1' not in hdr:
                continue  # dead module
            wcs = WCS(hdr)
            (col, row) = wcs.all_world2pix([skycoord.ra.deg], [skycoord.dec.deg], 0)
            col, row = col[0], row[0]
            # Dide the coordinate gall within the channel footprint?
            if (col > 0) and (col < hdr['NAXIS1']) and (row > 0) and (row < hdr['NAXIS2']):
                return channel, col, row
        return None, None, None

    def plot_cutout(self, skycoord, size=10, **kwargs):
        """Plots a small part of an FFI around a sky coordinate."""
        channel, col, row = self.skycoord_to_pixel(skycoord)
        extent = (int(col-size/2), int(col+size/2),
                  int(row-size/2), int(row+size/2))
        img = self.hdulist[channel].data[extent[2]: extent[3],
                                         extent[0]: extent[1]]
        plot_image(img, extent=extent, **kwargs)

    def plot(self):
        """Plot all channels on a grid."""
        pass
