FROM --platform=linux/amd64 nvidia/cuda:12.1.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-cu"]

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl git wget htop gnupg python3 python3-pip python3-venv python-is-python3

# install leanblueprint dependencies
RUN apt install -y python3-dev graphviz libgraphviz-dev pkg-config texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra texlive-xetex ghostscript

# install lean
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
RUN $HOME/.elan/bin/elan default stable

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
