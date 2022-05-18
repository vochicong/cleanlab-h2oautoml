CONDA ?= conda

all:
	jupytext --sync --pipe 'isort - --treat-comment-as-code "# %%" --float-to-top' --pipe black *.ipynb

env:
	$(CONDA) env update -n dev -f environment.yml
