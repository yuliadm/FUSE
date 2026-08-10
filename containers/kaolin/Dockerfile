# syntax=docker/dockerfile:1.7

# PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.4 and cuDNN 9 are already
# installed. The development image also provides nvcc for Kaolin CUDA code.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG KAOLIN_VERSION=0.18.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    HOME=/home/fuse \
    XDG_CACHE_HOME=/workspace/.cache/xdg \
    TORCH_HOME=/workspace/.cache/torch \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    JUPYTER_CONFIG_DIR=/workspace/.cache/jupyter \
    IPYTHONDIR=/workspace/.cache/ipython \
    PYTHONPATH=/workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# All operating-system installation runs as root before the runtime user exists.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Isolate Kaolin and notebook packages while reusing the PyTorch/CUDA stack
# already installed in the official base image.
RUN python -m venv --system-site-packages "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade \
        "pip==25.1.1" \
        "setuptools==75.8.0" \
        "wheel==0.45.1"

# Use NVIDIA's wheel built for this exact Torch and CUDA combination.
RUN python -m pip install --prefer-binary \
        "kaolin==${KAOLIN_VERSION}" \
        -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html"

# FUSE alignment, inspection and notebook dependencies.
RUN python -m pip install --prefer-binary \
        "numpy==1.26.4" \
        "scipy==1.14.1" \
        "open3d==0.18.0" \
        "trimesh==4.5.3" \
        "networkx==3.4.2" \
        "plotly==5.24.1" \
        "jupyterlab==4.3.5" \
        "ipykernel==6.29.5" \
        "ipywidgets==8.1.5"

# Fail the build immediately if the pinned environment is inconsistent.
RUN python -m pip check \
    && python -c "from importlib.metadata import version; import kaolin, open3d, torch, torchvision, trimesh; assert torch.__version__.startswith('2.5.1'); assert torchvision.__version__.startswith('0.20.1'); assert torch.version.cuda == '12.4'; assert version('kaolin') == '${KAOLIN_VERSION}'; print('FUSE Kaolin installation OK')" \
    && python -m jupyterlab --version

# Create the host-matched non-root identity only after privileged installation.
RUN groupadd --gid "${GROUP_ID}" fuse \
    && useradd \
        --uid "${USER_ID}" \
        --gid "${GROUP_ID}" \
        --create-home \
        --shell /bin/bash \
        fuse \
    && mkdir -p /workspace/data /workspace/notebooks /workspace/.cache \
    && chown -R fuse:fuse /workspace /home/fuse

USER fuse
WORKDIR /workspace

EXPOSE 8891

CMD ["/bin/bash"]
