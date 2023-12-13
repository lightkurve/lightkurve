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

# Make sure we use temporary directories for the config and cache
# so that the tests are insensitive to local configuration.

os.environ['XDG_CONFIG_HOME'] = tempfile.mkdtemp('lightkurve_config')
os.mkdir(os.path.join(os.environ['XDG_CONFIG_HOME'], 'lightkurve'))

# Let users optionally specify XDG_CACHE_HOME for a test run
# use case: in a local dev env, an user might want to reuse an existing dir for cache,
# so as to speed up remote-data tests
if os.environ.get('XDG_CACHE_HOME', '') == '':
    os.environ['XDG_CACHE_HOME'] = tempfile.mkdtemp('lightkurve_cache')
else:
    print(f"lightkurve conftest: Use user-specified XDG_CACHE_HOME: {os.environ['XDG_CACHE_HOME']}")

_cache_dir = os.path.join(os.environ['XDG_CACHE_HOME'], 'lightkurve')
if not os.path.isdir(_cache_dir):
    os.mkdir(_cache_dir)

_astropy_cache_dir = os.path.join(os.environ['XDG_CACHE_HOME'], 'astropy')
if not os.path.isdir(_astropy_cache_dir):
    os.mkdir(_astropy_cache_dir)
