CONDA ?= conda

ENV=dev
RUN=$(CONDA) run -n $(ENV)

all:
	$(RUN) \
	jupytext --sync --pipe 'isort - --treat-comment-as-code "# %%" --float-to-top' --pipe black *.ipynb

env:
	$(CONDA) env update -n $(ENV) -f environment.yml
