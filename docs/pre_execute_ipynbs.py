#
# Batch Pre-execute ipynbs for doc build, to workaround the issue that
# in standard nbsphinx build, when there is an error. e.g, timeout from MAST
# it'd halt the build. One has to start from scratch again.
#
# This script pre-execute the ipynbs in such a way that if there is an error
# in executing a notebook, it'd skip it and move on to the next one.
# Users can repeatedly invoke the script until all ipynbs are pre-executed.
# Unlike nbsphinx build, previously successfully executed notebooks won't be re-executed.
#
# The entire workflow is as follows:
"""
# A convenient local backup for just in case
rm -fr source/tutorials_original/
cp -r source/tutorials  source/tutorials_original/

rm -fr source/tutorials_pre_execute/

# Pre-execute ipynbs in source/tutorials, and store them in source/tutorials_pre_execute/
#   repeat as many times as needed until all are done
python pre_execute_ipynbs.py

# Once done, copy the pre-executed notebooks to original tutorials directory
cp -r source/tutorials_pre_execute/* source/tutorials/

# Now we can build the doc, with pre-executed notebooks
make html

# once done, revert the executed ipynbs in source/tutorials, as they should not be committed.
find source/tutorials -name "*.ipynb" | xargs git restore
"""

#
#  requires package papermill
#  https://papermill.readthedocs.io/
#
import glob
import os
import sys
import shutil

import papermill as pm

def batch_execute(ipynb_list):
    os.makedirs("source/tutorials_pre_execute/", exist_ok=True)
    num_entries = len(ipynb_list)
    num_failed = 0
    num_run = 0
    num_skipped = 0
    for i, in_file in enumerate(ipynb_list):
        try:
            out_file_final = in_file.replace("/tutorials/", "/tutorials_pre_execute/")
            out_file_tmp = out_file_final.replace(".ipynb", "_tmp.ipynb")

            out_dir = os.path.dirname(out_file_final)
            os.makedirs(out_dir, exist_ok=True)

            in_filename = os.path.basename(in_file)
            print(f"{i+1:2d}/{num_entries}. Process {in_filename}")
            if os.path.isfile(out_file_final):
                num_skipped += 1
                print(f"        {in_filename} has been processed. Skip it.")
                continue
            if "how-to-open-a-lightcurve-in-excel.ipynb" == in_filename :
                # Skip it: it'd fail, as the last cell assumes google colab environment
                num_skipped += 1
                print(f"        Skip {in_filename} (manual exclusion)")
                continue

            res = pm.execute_notebook(
                in_file,
                out_file_tmp,
                parameters=None,
                progress_bar=True,
            )
            shutil.move(out_file_tmp, out_file_final)
            num_run += 1
        except Exception as err:
            num_failed += 1
            print(f"Error in processing {i}: {type(err).__name__}: {err}", file=sys.stderr, flush=True)

    print(f"Executed: {num_run} ; Skipped: {num_skipped} ; Failed: {num_failed} ")
    return dict(num_run=num_run, num_skipped=num_skipped, num_failed=num_failed)


if __name__ == "__main__":
    ipynb_list =  glob.glob("./source/tutorials/**/*.ipynb")
    # ipynb_list = [
    #     "source/tutorials/1-getting-started/how-to-open-a-lightcurve-in-excel.ipynb",
    #     "source/tutorials/1-getting-started/interactively-inspecting-data.ipynb",
    # ]
    run_summary = batch_execute(ipynb_list)
    if run_summary["num_failed"] > 0:
        exit(1)  # signify there have been errors
    else:
        exit(0)
