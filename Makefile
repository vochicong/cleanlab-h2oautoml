CONDA ?= conda

all:
	jupytext --sync --pipe black *.ipynb

env:
	$(CONDA) env update -n dev -f environment.yml