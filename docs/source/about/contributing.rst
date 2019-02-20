.. _contributing:

======================================
Reporting issues and proposing changes
======================================

Lightkurve is actively developed on its `GitHub repository <https://github.com/KeplerGO/lightkurve>`_.

If you encounter a problem with Lightkurve, we encourage you to
`open an issue on the GitHub repository <https://github.com/KeplerGO/lightkurve/issues>`_
or to e-mail the Kepler/K2 Guest Observer Office at keplergo@mail.arc.nasa.gov.

If you would like to contribute a patch for a bugfix, please go ahead and open a pull request.


Proposing changes to lightkurve using GitHub pull requests
----------------------------------------------------------

We welcome suggestions for enhancements or new features to Lightkurve via GitHub.

If you want to make a significant change such as adding a new feature, we recommend opening a GitHub issue to discuss the changes.
Once you are ready to propose the changes, please go ahead and open a pull request.

If in doubt on how to open a pull request, we recommend Astropy's
"`How to make a code contribution <http://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`_" tutorial.
In brief, the steps are as follows:

1. Fork the main **lightkurve** repository by logging into GitHub, browsing to
   ``https://github.com/KeplerGO/lightkurve`` and clicking on ``Fork`` in the top right corner.

2. Clone your fork to your computer:

.. code-block:: bash

    $ git clone https://github.com/YOUR-GITHUB-USERNAME/lightkurve.git

3. Install the development version of lightkurve:

.. code-block:: bash

    $ cd lightkurve
    $ pip install -e .

4. Add the KeplerGO remote to your GitHub enviroment:

.. code-block:: bash

    $ git remote add upstream https://github.com/KeplerGO/lightkurve.git

5. Let's make sure everything is setup correctly. Execute:

.. code-block:: bash

    $ git remote -v

You should see something like this:

.. code-block:: bash

    origin	https://github.com/YOUR-GITHUB-USERNAME/lightkurve.git (fetch)
    origin	https://github.com/YOUR-GITHUB-USERNAME/lightkurve.git (push)
    upstream	https://github.com/KeplerGO/lightkurve.git (fetch)
    upstream	https://github.com/KeplerGO/lightkurve.git (push)

6. Now you are ready to start contributing; make a new branch with a name of your choice and checkout:

.. code-block:: bash

    $ git branch name-of-your-branch
    $ git checkout name-of-your-branch

7. Do the changes you want and add them:

.. code-block:: bash

    $ git add FILE-YOU-ADDED-OR-MODIFIED

8. Commit and push your changes:

.. code-block:: bash

    $ git commit -m "description of changes"
    $ git push origin name-of-your-branch

9. Head to https://github.com/KeplerGO/lightkurve and you should now see a button
   "Compare and open a pull request".  Click the button and submit your pull request.


10. That's it! :)
