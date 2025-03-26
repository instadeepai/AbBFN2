ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    g++ \
    wget \
    unzip \
    curl \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install Micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/abbfn2/bin:$PATH

# Install conda dependencies
COPY environment.yaml /tmp/environment.yaml

ARG ACCELERATOR

# Create environment and install JAX
RUN micromamba create -y --file /tmp/environment.yaml \
    && micromamba install -y -n abbfn2 -c conda-forge jax==0.4.34 jaxlib==0.4.34 \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete

ENV PATH=/opt/conda/envs/abbfn2/bin/:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/abbfn2/lib/:$LD_LIBRARY_PATH

COPY . /app

# Create main working folder
# RUN mkdir /app
WORKDIR /app
RUN pip install -U "huggingface_hub[cli]"

# Disable debug, info, and warning tensorflow logs
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_CPP_MIN_LOG_LEVEL=3