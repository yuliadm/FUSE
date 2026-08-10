# PyTorch, torchvision, torchaudio, CUDA 12.4 and cuDNN 9 are preinstalled in
# this official image. The project venv inherits those packages instead of
# downloading the multi-gigabyte PyTorch wheel set again.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG HUNYUAN_REF=82920d643c0dc2f7bfd7255f45f62d386edfe60c

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    U2NET_HOME=/workspace/.cache/rembg \
    PYTHONPATH=/opt/Hunyuan3D-2.1/hy3dshape:/workspace \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

# All operating-system installation must happen while the build runs as root.
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        libglu1-mesa \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Use an isolated project environment while retaining Torch/CUDA from the
# official base image through --system-site-packages.
RUN /opt/conda/bin/python -m venv --system-site-packages "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade \
        "pip==25.1.1" \
        "setuptools==75.8.0" \
        "wheel==0.45.1"

# Copy dependency manifests before cloning source so dependency layers remain
# reusable when only the pinned Hunyuan source revision changes.
COPY requirements-shape.txt /tmp/requirements-shape.txt
COPY requirements-notebook.txt /tmp/requirements-notebook.txt

RUN python -m pip install --prefer-binary \
        -r /tmp/requirements-shape.txt

RUN python -m pip install --prefer-binary \
        -r /tmp/requirements-notebook.txt

# Pin the source revision so every teammate builds the same Hunyuan code.
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git \
        /opt/Hunyuan3D-2.1 \
    && git -C /opt/Hunyuan3D-2.1 checkout --detach "${HUNYUAN_REF}"

# Fail during the image build—not in a teammate's notebook—if the Python
# environment, Hunyuan pipeline, or JupyterLab entry point is inconsistent.
# requirements-shape.txt replaces the base image's Ninja 1.11.1.1 package,
# whose wheel metadata is incompatible with recent pip versions.
RUN python -m pip check \
    && python -c "from importlib.metadata import version; import cv2, jupyterlab, pymeshlab, torch, torchvision, trimesh; from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline; from hy3dshape.rembg import BackgroundRemover; assert version('ninja') == '1.11.1.4'; assert torch.__version__.startswith('2.5.1'); assert torchvision.__version__.startswith('0.20.1'); assert torch.version.cuda == '12.4'; print('FUSE Hunyuan shape installation OK')" \
    && python -m jupyterlab --version

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
