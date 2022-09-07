# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/python-3-miniconda/.devcontainer/base.Dockerfile

# FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

FROM condaforge/mambaforge
ENV CONDA=mamba

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
# ARG NODE_VERSION="none"
# RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends make

# RUN conda install conda-forge::mamba -n base
COPY environment.yml /tmp/conda-tmp/
RUN umask 0002 && mamba env create -n dev -f /tmp/conda-tmp/environment.yml

# [Optional] Uncomment to install a different version of Python than the default
# RUN conda install -y python=3.6 \
#     && pip install --no-cache-dir pipx \
#     && pipx reinstall-all
