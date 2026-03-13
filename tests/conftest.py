import os
import tempfile
import json
import urllib.parse
from vcr.util import read_body

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


# ---------------------------------------------------------------------------
# pytest-recording (vcrpy) configuration
# ---------------------------------------------------------------------------

def vcr_body_matcher(r1, r2):
    """Custom body matcher that ignores 'cacheBreaker' in MAST API requests.
    Also safely handles None bodies.
    """
    b1 = read_body(r1)
    b2 = read_body(r2)

    if b1 == b2:
        return True

    if b1 is None or b2 is None:
        return False

    try:
        # MAST API uses URL-encoded JSON in a 'request' parameter
        # We try to decode and ignore 'cacheBreaker'
        q1 = urllib.parse.parse_qs(b1.decode("utf-8"))
        q2 = urllib.parse.parse_qs(b2.decode("utf-8"))

        if "request" in q1 and "request" in q2:
            req1 = json.loads(q1["request"][0])
            req2 = json.loads(q2["request"][0])

            # Ignore cacheBreaker
            req1.pop("cacheBreaker", None)
            req2.pop("cacheBreaker", None)

            return req1 == req2
    except (UnicodeDecodeError, AttributeError, json.JSONDecodeError, IndexError, TypeError):
        pass

    return b1 == b2


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR for lightkurve tests.

    - cassette_library_dir: store cassettes alongside tests
    - record_mode: controlled by the --record-mode command-line flag
    - match_on: MAST API uses POST with JSON bodies, so we match on method, scheme,
      host, port, path, query, and our custom 'body' matcher.
    - decode_compressed_response: ensures gzipped responses are stored readable
    """
    return {
        "cassette_library_dir": os.path.join(os.path.dirname(__file__), "cassettes"),
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
        "decode_compressed_response": True,
    }


def pytest_recording_configure(config, vcr):
    """Register custom matchers with vcrpy."""
    vcr.register_matcher("body", vcr_body_matcher)


@pytest.fixture(autouse=True)
def clear_search_memoization_cache():
    """Clear the in-memory memoization cache on search functions after each test.

    search_lightcurve(), search_targetpixelfile(), and search_tesscut() are
    decorated with @cached (from the `memoization` package), which caches results
    in memory. If the cache isn't cleared between tests, subsequent calls with
    the same arguments return the cached result without making an HTTP request,
    preventing vcrpy from intercepting and recording those requests.
    """
    yield  # run the test first
    try:
        from lightkurve.search import (
            search_lightcurve,
            search_targetpixelfile,
            search_tesscut,
        )
        for fn in (search_lightcurve, search_targetpixelfile, search_tesscut):
            if hasattr(fn, 'cache_clear'):
                fn.cache_clear()
    except ImportError:
        pass
