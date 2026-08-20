# syntax=docker/dockerfile:1.7

# FUSE Stage 0: mask-conditioned intact-reference hypothesis generation with
# FLUX.1 Fill [dev]. The transformer and T5-XXL text encoder are loaded in
# bitsandbytes NF4 4-bit by the notebook; PyTorch/VAEs remain in bfloat16.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    HOME=/home/fuse \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    HF_XET_CACHE=/workspace/.cache/huggingface/xet \
    HF_XET_HIGH_PERFORMANCE=0 \
    HF_XET_NUM_CONCURRENT_RANGE_GETS=4 \
    HF_HUB_DOWNLOAD_TIMEOUT=600 \
    HF_HUB_ETAG_TIMEOUT=60 \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache/xdg \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    JUPYTER_CONFIG_DIR=/workspace/.cache/jupyter \
    IPYTHONDIR=/workspace/.cache/ipython \
    BITSANDBYTES_NOWELCOME=1 \
    PYTHONPATH=/workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Keep an isolated notebook environment while reusing CUDA-enabled PyTorch
# from the base image. Versions are pinned so a later package release cannot
# silently change the quantization or Fill pipeline behavior.
RUN python -m venv --system-site-packages "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade \
        "pip==25.1.1" \
        "setuptools==75.8.0" \
        "wheel==0.45.1"

RUN python -m pip install --prefer-binary \
        "diffusers==0.39.0" \
        "transformers==4.57.6" \
        "accelerate==1.10.1" \
        "bitsandbytes==0.50.1" \
        "huggingface-hub==0.34.4" \
        "safetensors==0.8.0" \
        "sentencepiece==0.2.1" \
        "protobuf==6.32.0" \
        "hf-xet==1.5.2" \
        "numpy==1.26.4" \
        "Pillow==11.3.0" \
        "matplotlib==3.10.5" \
        "pandas==2.3.1" \
        "psutil==7.0.0" \
        "jupyterlab==4.4.5" \
        "ipykernel==6.30.1" \
        "ipywidgets==8.1.7"

# Catch packaging/import failures during the image build. Model weights are
# deliberately not downloaded here because FLUX.1 Fill is gated and requires
# the user's own accepted Hugging Face token at runtime.
RUN python -m pip check \
    && python -c "import torch, bitsandbytes; from diffusers import BitsAndBytesConfig, FluxFillPipeline, FluxTransformer2DModel; from transformers import T5EncoderModel; assert torch.version.cuda is not None; print('FUSE FLUX.1 Fill 4-bit runtime OK')" \
    && python -m jupyterlab --version

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

EXPOSE 8895

CMD ["/bin/bash"]
