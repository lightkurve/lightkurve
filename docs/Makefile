# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
#SPHINXOPTS    = -j auto    # PARALLEL BUILDS DO NOT WORK IN RECENT SPHINX VERSIONS DUE TO A SECURITY ISSUE
SPHINXOPTS    =
SPHINXBUILD   = poetry run python -m sphinx
SPHINXPROJ    = lightkurve
SOURCEDIR     = source
BUILDDIR      = ../../lightkurve-docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	cp ../pyproject.toml $(BUILDDIR)/html

execute-notebooks:
	poetry run python pre_execute_ipynbs.py

clear-notebook-cache:
	rm -rf source/tutorials_pre_execute/*

sync-notebooks:
	# Do some rsync magic
	rsync -av --relative --remove-source-files source/tutorials_pre_execute/./*/*.ipynb source/tutorials

notebooks:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D nbsphinx_execute='false'

# ghp-import can be installed with pip as pip install ghp-import
upload:
	ghp-import -m "Update docs" $(BUILDDIR)/html
	git push origin gh-pages

clean:
	# Remove previous output
	-rm -rf $(BUILDDIR)/*
	# Force autogenerating the API docs
	-rm source/reference/api/*
