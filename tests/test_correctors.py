import pytest


@pytest.mark.remote_data
def test_to_corrector():
    """Does the tpf.to_corrector('pld') convenience method work?"""
    from lightkurve import KeplerTargetPixelFile
    from .test_targetpixelfile import TABBY_TPF

    tpf = KeplerTargetPixelFile(TABBY_TPF)
    lc = tpf.to_corrector("pld").correct()
    assert len(lc.flux) == len(tpf.time)
