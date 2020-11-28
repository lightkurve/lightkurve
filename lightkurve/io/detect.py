"""Provides a function to automatically detect Kepler/TESS file types."""
from astropy.io.fits import HDUList


__all__ = ['detect_filetype']


def detect_filetype(hdulist: HDUList) -> str:
    """Returns Kepler and TESS file types given a FITS object.

    This function will detect the file type by looking at both the TELESCOP and
    CREATOR keywords in the first extension of the FITS header. If the file is
    recognized as a Kepler or TESS data product, one of the following strings
    will be returned:

        * `'KeplerTargetPixelFile'`
        * `'TessTargetPixelFile'`
        * `'KeplerLightCurve'`
        * `'TessLightCurve'`
        * `'K2SFF'`
        * `'EVEREST'`
        * `'K2SC'`
        * `'K2VARCAT'`
        * `'QLP'`

    If the data product cannot be detected, `None` will be returned.

    Parameters
    ----------
    hdulist : astropy.io.fits.HDUList object
        A FITS file.

    Returns
    -------
    filetype : str or None
        A string describing the detected filetype. If the filetype is not
        recognized, `None` will be returned.
    """
    # Is it a MIT/QLP TESS FFI Quicklook Pipeline light curve?
    # cf. http://archive.stsci.edu/hlsp/qlp
    if "mit/qlp" in hdulist[0].header.get('origin', '').lower():
        return "QLP"

    # Is it a K2VARCAT file?
    # There are no self-identifying keywords in the header, so go by filename.
    if "hlsp_k2varcat" in (hdulist.filename() or ""):
        return "K2VARCAT"

    # Is it a K2SC file?
    if "k2sc" in hdulist[0].header.get('creator', '').lower():
        return "K2SC"

    # Is it a K2SFF file?
    try:
        # There are no metadata keywords identifying K2SFF FITS files,
        # so we go by structure.
        if hdulist[1].header.get('EXTNAME') == "BESTAPER" and hdulist[1].header.get("TTYPE4") == "ARCLENGTH":
            return "K2SFF"
    except Exception:
        pass

    # Is it an EVEREST file?
    try:
        if "EVEREST" in str(hdulist[0].header.get('COMMENT')):
            return "EVEREST"
    except Exception:
        pass

    # Is it an official data product?
    header = hdulist[0].header
    try:
        # use `telescop` keyword to determine mission
        # and `creator` to determine tpf or lc
        if 'TELESCOP' in header.keys():
            telescop = header['telescop'].lower()
        else:
            # Some old custom TESS data did not define the `TELESCOP` card
            telescop = header['mission'].lower()
        creator = header['creator'].lower()
        origin = header['origin'].lower()
        if telescop == 'kepler':
            # Kepler TPFs will contain "TargetPixelExporterPipelineModule"
            if 'targetpixel' in creator:
                return 'KeplerTargetPixelFile'
            # Kepler LCFs will contain "FluxExporter2PipelineModule"
            elif ('fluxexporter' in creator or 'lightcurve' in creator
                or 'lightcurve' in creator):
                return 'KeplerLightCurve'
        elif telescop == 'tess':
            # TESS TPFs will contain "TargetPixelExporterPipelineModule"
            if 'targetpixel' in creator:
                return 'TessTargetPixelFile'
            # TESS LCFs will contain "LightCurveExporterPipelineModule"
            elif 'lightcurve' in creator:
                return 'TessLightCurve'
            # Early versions of TESScut did not set a good CREATOR keyword
            elif 'stsci' in origin:
                return 'TessTargetPixelFile'
    # If the TELESCOP or CREATOR keywords don't exist we expect a KeyError;
    # if one of them is Undefined we expect `.lower()` to yield an AttributeError.
    except (KeyError, AttributeError):
        return None
