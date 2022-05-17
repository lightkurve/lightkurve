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
        cfg_dest_path = Path(lk.config.get_config_dir(), 'lightkurve.cfg')
        cfg_src_path = get_pkg_data_filename("data/lightkurve_sr_cols_added.cfg")
        shutil.copy(cfg_src_path, cfg_dest_path)
        lk.conf.reload()
        assert ['proposal_id'] == lk.conf.search_result_display_extra_columns
    finally:
        # cleanup: remvoe the custom config file and its effect
        cfg_dest_path.unlink()
        lk.conf.reload()

