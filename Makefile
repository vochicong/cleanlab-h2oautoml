CONDA ?= conda

all:
	$(CONDA) run -n dev jupytext --sync --pipe black *.ipynb
