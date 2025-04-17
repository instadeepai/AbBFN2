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
COPY . /app

ARG ACCELERATOR
RUN if [ "${ACCELERATOR}" = "GPU" ]; then \
    sed -i 's/jax==/jax[cuda12_pip]==/g' /app/environment.yaml && \
    sed -i 's/libtpu_releases\.html/jax_cuda_releases\.html/g' /app/environment.yaml;\
    echo "    - nvidia-cudnn-cu12==8.9.7.29" >> /app/environment.yaml; fi

RUN if [ "${ACCELERATOR}" = "TPU" ]; then \
    sed -i 's/jax==/jax[tpu]==/g' /app/environment.yaml; fi

RUN if [ "${ACCELERATOR}" = "CPU" ]; then \
    echo "Building for cpu" ; fi

# Create environment
RUN micromamba create -y --file /app/environment.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete

ENV PATH=/opt/conda/envs/abbfn2/bin/:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/abbfn2/lib/:$LD_LIBRARY_PATH

# Create main working folder
WORKDIR /app
