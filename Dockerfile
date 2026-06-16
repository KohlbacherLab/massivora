# syntax=docker/dockerfile:1
#
# Production GPU image for Massivora (CUDA-enabled).
#
#   docker build -t massivora:gpu .
#   docker run --rm --gpus all massivora:gpu massivora --help
#
# Multi-stage: build the native CUDA extension against a CUDA *devel* image,
# then ship only the conda env on a slim CUDA *runtime* image.
# A CPU-only host can still run this image (GPU is auto-detected as optional),
# but for a smaller CPU-only image use Dockerfile.cpu instead.

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

############################
# Stage 1: build
############################
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl bzip2 ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Install micromamba (fast conda-compatible package manager).
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xvj -C /usr/local bin/micromamba

COPY . /opt/massivora-src
WORKDIR /opt/massivora-src

# Create the conda env with all native + runtime deps (nlopt, eigen, ost, ...).
RUN micromamba create -y -p ${MAMBA_ROOT_PREFIX}/envs/massivora -f environment.yml && \
    micromamba clean --all --yes

# Install CuPy via conda, matched to the base image's CUDA series (12.x).
RUN micromamba install -y -p ${MAMBA_ROOT_PREFIX}/envs/massivora \
        -c conda-forge cupy "cuda-version=12" && \
    micromamba clean --all --yes

# Build & install the native extension with CUDA enabled.
# `micromamba run` activates the env so CMake picks up the conda compilers,
# Eigen and NLopt (a bare PATH tweak does not run the activation scripts).
RUN micromamba run -p ${MAMBA_ROOT_PREFIX}/envs/massivora \
        pip install . --no-build-isolation --no-deps \
        -C cmake.define.ENABLE_CUDA=ON

############################
# Stage 2: runtime
############################
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG MAMBA_ROOT_PREFIX=/opt/conda

COPY --from=build ${MAMBA_ROOT_PREFIX}/envs/massivora ${MAMBA_ROOT_PREFIX}/envs/massivora
ENV PATH=${MAMBA_ROOT_PREFIX}/envs/massivora/bin:$PATH

# Expose all GPUs to the container at runtime (used with `--gpus all`).
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /work
ENTRYPOINT ["massivora"]
CMD ["--help"]
