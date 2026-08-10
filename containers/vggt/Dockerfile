FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG VGGT_REF=a288dd0f14786c93483e45524328726ab7b1b4ce
ARG LIGHTGLUE_REF=2f23ca2ea9638cecad7f7220795210fc6b8353c3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    U2NET_HOME=/workspace/.cache/rembg \
    PYTHONPATH=/opt/vggt:/workspace \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

# All operating-system installation must happen while the build runs as root.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Keep Python dependencies isolated from Ubuntu's system Python.
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# VGGT officially pins Torch 2.3.1 / Torchvision 0.18.1. Their newest
# supported CUDA wheel is cu121; it runs on the CUDA 12.4 base/driver stack.
RUN python -m pip install --no-cache-dir \
        torch==2.3.1 \
        torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Pin the source revision so every teammate builds the same VGGT code.
RUN git clone https://github.com/facebookresearch/vggt.git /opt/vggt \
    && git -C /opt/vggt checkout --detach "${VGGT_REF}"

WORKDIR /opt/vggt

# Install both VGGT's declared runtime dependencies and VGGT itself.
# Do not suppress failures: a broken VGGT install must fail the image build.
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --editable .

# Dependencies used by the official demo_colmap.py bundle-adjustment path.
# LightGlue is pinned because the upstream requirements file otherwise follows
# the moving head of its Git repository.
RUN python -m pip install --no-cache-dir \
        hydra-core \
        matplotlib \
        omegaconf \
        pyceres==2.3 \
        pycolmap==3.10.0 \
        pydantic==2.10.6 \
        requests \
        scipy \
        tqdm \
        trimesh \
    && python -m pip install --no-cache-dir \
        "lightglue @ git+https://github.com/jytime/LightGlue.git@${LIGHTGLUE_REF}"

# FUSE Stage 1 notebook, point-cloud, and background-removal utilities.
RUN python -m pip install --no-cache-dir \
        ipykernel \
        ipywidgets \
        jupyterlab \
        open3d \
        onnxruntime-gpu \
        pandas \
        plotly \
        pymeshlab \
        rembg \
        scikit-image

# Fail during the build—not during a teammate's notebook—if the essential
# Python packages are unavailable.
RUN python -c "import lightglue, pyceres, pycolmap, torch, torchvision, vggt; print('VGGT installation OK')"

# Create the non-root runtime identity only after all privileged installation.
RUN groupadd --gid "${GROUP_ID}" fuse \
    && useradd \
        --uid "${USER_ID}" \
        --gid "${GROUP_ID}" \
        --create-home \
        --shell /bin/bash \
        fuse \
    && mkdir -p /workspace/data /workspace/notebooks /workspace/.cache \
    && chown -R fuse:fuse /workspace /home/fuse

ENV HOME=/home/fuse

USER fuse
WORKDIR /workspace

CMD ["/bin/bash"]
