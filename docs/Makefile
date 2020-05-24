# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -j auto
SPHINXBUILD   = sphinx-build
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

notebooks:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D nbsphinx_execute='always'

# ghp-import can be installed with pip as pip install ghp-import
upload:
	ghp-import -m "Update docs" $(BUILDDIR)/html
	git push origin gh-pages

clean:
	# Remove previous output
	-rm -rf $(BUILDDIR)/*
	# Force autogenerating the API docs
	-rm source/api/lightkurve.*
