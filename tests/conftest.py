import os
import tempfile

import pytest


def pytest_runtest_setup(item):
    r"""Our tests will often run in headless virtual environments. For this
    reason, we enforce the use of matplotlib's robust Agg backend, because it
    does not require a graphical display.

    This avoids errors such as:
        c:\hostedtoolcache\windows\python\3.7.5\x64\lib\tkinter\__init__.py:2023: TclError
        This probably means that tk wasn't installed properly.
    """
    import matplotlib

    matplotlib.use("Agg")

# Add a marker @pytest.mark.memtest
# - used to mark tests that stress memory, typically done by limiting the memory Python can use
# - thus they should be run in isolation.
#
# - skipped by default
# - tests marked as such can be run by "-m memtest" option


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "memtest: mark memory usage tests that need to be run in isolation"
    )


def pytest_collection_modifyitems(config, items):
    keywordexpr = config.option.keyword
    markexpr = config.option.markexpr
    if keywordexpr or markexpr:
        return  # let pytest handle this

    skip_memtest = pytest.mark.skip(reason='memtest skipped, need -m memtest option to run')
    for item in items:
        if 'memtest' in item.keywords:
            item.add_marker(skip_memtest)

# Make sure we use temporary directories for the config
# so that the tests are insensitive to local configuration.

os.environ['XDG_CONFIG_HOME'] = tempfile.mkdtemp('lightkurve_config')
os.mkdir(os.path.join(os.environ['XDG_CONFIG_HOME'], 'lightkurve'))