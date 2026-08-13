# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm

ARG USER_ID=1000
ARG GROUP_ID=1000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    HOME=/home/fuse \
    XDG_CACHE_HOME=/workspace/.cache/xdg \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    FUSE_ROOT=/workspace \
    PYTHONPATH=/workspace/app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        passwd \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade \
        "pip==25.1.1" \
        "setuptools==75.8.0" \
        "wheel==0.45.1"

COPY requirements.txt /opt/fuse-app/requirements.txt
RUN python -m pip install --prefer-binary -r /opt/fuse-app/requirements.txt \
    && python -m pip check \
    && python -c "import numpy, open3d, pandas, plotly, streamlit, trimesh; print('FUSE app installation OK')"

# Reuse UID/GID 1000 when a base image already defines them; otherwise create
# the host-matched identity. This avoids duplicate UID/GID build failures.
RUN set -eux; \
    existing_group="$(getent group "${GROUP_ID}" | cut -d: -f1 || true)"; \
    if [[ -z "${existing_group}" ]]; then \
        groupadd --gid "${GROUP_ID}" fuse; \
    fi; \
    existing_user="$(getent passwd "${USER_ID}" | cut -d: -f1 || true)"; \
    if [[ -z "${existing_user}" ]]; then \
        useradd \
            --uid "${USER_ID}" \
            --gid "${GROUP_ID}" \
            --create-home \
            --shell /bin/bash \
            fuse; \
    elif [[ "${existing_user}" != "fuse" ]]; then \
        usermod \
            --login fuse \
            --home /home/fuse \
            --shell /bin/bash \
            --gid "${GROUP_ID}" \
            "${existing_user}"; \
    fi; \
    mkdir -p /home/fuse /workspace/app /workspace/data /workspace/.cache; \
    chown -R "${USER_ID}:${GROUP_ID}" /workspace /home/fuse

USER fuse
WORKDIR /workspace/app

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "/workspace/app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
