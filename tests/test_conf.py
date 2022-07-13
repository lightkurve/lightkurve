from astropy.utils.data import get_pkg_data_filename

import os
from pathlib import Path
import shutil
import tempfile

import lightkurve as lk


def test_read_conf_from_file():
    """Sanity test to ensure lightkurve per-user config is in the expected location."""

    # assert the default config
    assert [] == lk.conf.search_result_display_extra_columns
    # Use a custom config file, and assert the changes are read.
    try:
        use_custom_config_file("data/lightkurve_sr_cols_added.cfg")
        assert ['proposal_id'] == lk.conf.search_result_display_extra_columns
    finally:
        # cleanup: remove the custom config file and its effect
        remove_custom_config()


def use_custom_config_file(cfg_filepath):
    """Copy the config file in the given path (in tests) to the default lightkurve config file """
    cfg_dest_path = Path(lk.config.get_config_dir(), 'lightkurve.cfg')
    cfg_src_path = get_pkg_data_filename(cfg_filepath)
    shutil.copy(cfg_src_path, cfg_dest_path)
    lk.conf.reload()


def remove_custom_config():
    cfg_dest_path = Path(lk.config.get_config_dir(), 'lightkurve.cfg')
    cfg_dest_path.unlink()
    lk.conf.reload()


def test_get_cache_dir():
    # Sanity test default download dir
    # We can't meaningful assert the location for typical cases.
    # Because in test environment, it is overriden by XDG_CACHE_HOME env var
    # (typically to some temp location)
    actual_dir = lk.config.get_cache_dir()
    assert os.path.isdir(actual_dir)

    # Test customized default download dir
    with tempfile.TemporaryDirectory() as expected_base:
        try:
            # I want to test that the impl would create a dir if not there
            expected_dir = os.path.join(expected_base, "some_subdir")
            lk.conf.cache_dir = expected_dir
            actual_dir = lk.config.get_cache_dir()
            assert expected_dir == actual_dir
            assert os.path.isdir(actual_dir)

            # repeated calls would work
            # (e.g., it won't raise errors in attempting to mkdir for an existing dir)
            actual_dir = lk.config.get_cache_dir()
            assert expected_dir == actual_dir
        finally:
            lk.conf.cache_dir = None
