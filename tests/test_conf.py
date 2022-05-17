from astropy.utils.data import get_pkg_data_filename

from pathlib import Path
import shutil

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
        # cleanup: remvoe the custom config file and its effect
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
