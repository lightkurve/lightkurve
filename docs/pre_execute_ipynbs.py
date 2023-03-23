#
# Batch pre-execute ipynbs for doc build, to workaround the issue that
# in standard nbsphinx build, when there is an error, e.g, timeout from MAST
# it'd halt the build. One has to start from scratch again.
#
# This script pre-executes the ipynbs in such a way that if there is an error
# in executing a notebook, it'd skip it and move on to the next one.
# Users can repeatedly invoke the script until all ipynbs are pre-executed.
# Unlike nbsphinx build, previously successfully executed notebooks won't be re-executed.
#
# The entire workflow is as follows, using the Makefile:
"""
# Execute the notebooks, repeat this command until they are all compiled
# make execute-notebooks

# Move the notebooks in `tutorials_pre_execute` to `tutorials`
# make sync-notebooks

# If you need to clear the files in `tutorials_pre_execute` use
# make clear-notebooks
"""

#
#  requires package papermill
#  https://papermill.readthedocs.io/
#
import glob
import os
import sys
import shutil
import logging
from rich.logging import RichHandler

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import papermill as pm

def batch_execute(ipynb_list):
    os.makedirs("source/tutorials_pre_execute/", exist_ok=True)
    num_entries = len(ipynb_list)
    num_failed = 0
    num_run = 0
    num_skipped = 0
    num_finished = 0
    for i, in_file in tqdm(enumerate(ipynb_list), leave=True, position=0, total=num_entries, desc='Pre-Running Notebooks'):
        try:
            out_file_final = in_file.replace("/tutorials/", "/tutorials_pre_execute/")
            out_file_tmp = out_file_final.replace(".ipynb", "_tmp.ipynb")

            out_dir = os.path.dirname(out_file_final)
            os.makedirs(out_dir, exist_ok=True)

            in_filename = os.path.basename(in_file)
            log.debug(f"{i+1:2d}/{num_entries}. Process {in_filename}")
            if os.path.isfile(out_file_final):
                num_finished += 1
                log.debug(f"        {in_filename} has been processed. Skip it.")
                continue
            if "how-to-open-a-lightcurve-in-excel.ipynb" == in_filename :
                # Skip it: it'd fail, as the last cell assumes google colab environment
                num_skipped += 1
                log.warning(f"        Skip {in_filename} (manual exclusion)")
                continue

            log.debug(f'Executing {in_filename}')
            res = pm.execute_notebook(
                in_file,
                out_file_tmp,
                parameters=None,
                progress_bar=False,
            )
            log.debug(f'Executed {in_filename}')
            shutil.move(out_file_tmp, out_file_final)
            log.debug(f'Moved {out_file_tmp} to {out_file_final}')
            num_run += 1
        except Exception as err:
            num_failed += 1
            log.warning(f"Error in processing {i}: {type(err).__name__}: {err}", file=sys.stderr, flush=True)

    log.info(f"Executed: {num_run}/{num_entries} ; Previous Compiled: {num_finished}/{num_entries} ; Skipped: {num_skipped}/{num_entries} ; Failed: {num_failed}/{num_entries} ")
    if (num_run + num_finished + num_skipped) == num_entries:
        log.info(":red_heart-emoji:  [bold green]All Notebooks Completed[/] :red_heart-emoji:")
    else:
        log.warning(":broken_heart:  [bold red]Some notebooks failed. If this is due to a MAST time out, you should rerun this function (e.g. make execute-notebooks).[\] :broken_heart:")
    return dict(num_run=num_run, num_skipped=num_skipped, num_finished=num_finished, num_failed=num_failed)


if __name__ == "__main__":
#    logging.basicConfig(format='%(message)s', datefmt="[%X]")
    
    log = logging.getLogger('NOTEBOOKS')
    log.addHandler(RichHandler(markup=True))
    log.setLevel('INFO')

    ipynb_list =  glob.glob("./source/tutorials/**/*.ipynb")
    # ipynb_list = [
    #     "source/tutorials/1-getting-started/how-to-open-a-lightcurve-in-excel.ipynb",
    #     "source/tutorials/1-getting-started/interactively-inspecting-data.ipynb",
    # ]
    log.info("[bold green]Pre Compiling Notebooks[/]")
    run_summary = batch_execute(ipynb_list)
    if run_summary["num_failed"] > 0:
        exit(1)  # signify there have been errors
    else:
        exit(0)
