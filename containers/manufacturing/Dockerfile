# syntax=docker/dockerfile:1.7

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG TARGETARCH=amd64
ARG FREECAD_VERSION=1.1.3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    HOME=/home/fuse \
    XDG_CACHE_HOME=/workspace/.cache/xdg \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    JUPYTER_CONFIG_DIR=/workspace/.cache/jupyter \
    IPYTHONDIR=/workspace/.cache/ipython \
    QT_QPA_PLATFORM=offscreen \
    FREECAD_USER_HOME=/workspace/.cache/freecad \
    FUSE_FREECAD_CMD=/opt/freecad/AppRun \
    PYTHONPATH=/opt/fuse-manufacturing:/workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install operating-system dependencies while the build still runs as root.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libfreetype6 \
        libgl1 \
        libx11-6 \
        libxcb-cursor0 \
        libxcb1 \
        libxext6 \
        libxi6 \
        libxkbcommon0 \
        libxrender1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Use the current stable upstream FreeCAD bundle rather than Ubuntu's older
# package. Asset names and checksums are pinned so a changed download fails.
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64) \
        asset="FreeCAD_${FREECAD_VERSION}-Linux-x86_64-py311.AppImage"; \
        checksum="3a853eb69ee595f779f2255dbf80a765926981d8ff68903cefee4dfb03a8f5ef" ;; \
      arm64) \
        asset="FreeCAD_${FREECAD_VERSION}-Linux-aarch64-py311.AppImage"; \
        checksum="9a8f9f7f2802bb856f2bb70f53d536e2ae06569f4e6d718407803076104ff55e" ;; \
      *) echo "Unsupported TARGETARCH=${TARGETARCH}" >&2; exit 2 ;; \
    esac; \
    curl --fail --location --retry 5 \
      "https://github.com/FreeCAD/FreeCAD/releases/download/${FREECAD_VERSION}/${asset}" \
      --output /tmp/FreeCAD.AppImage; \
    echo "${checksum}  /tmp/FreeCAD.AppImage" | sha256sum --check -; \
    chmod +x /tmp/FreeCAD.AppImage; \
    cd /opt; \
    /tmp/FreeCAD.AppImage --appimage-extract >/dev/null; \
    mv /opt/squashfs-root /opt/freecad; \
    rm /tmp/FreeCAD.AppImage

RUN /opt/freecad/AppRun freecadcmd --version

# Match the other FUSE containers: one isolated venv containing the notebook
# environment and all stage-specific Python dependencies.
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade \
        "pip==25.1.1" \
        "setuptools==75.8.0" \
        "wheel==0.45.1"

COPY requirements.txt /opt/fuse-manufacturing/requirements.txt
RUN python -m pip install --prefer-binary \
        -r /opt/fuse-manufacturing/requirements.txt

COPY fuse_manufacturing /opt/fuse-manufacturing/fuse_manufacturing
COPY freecad /opt/fuse-manufacturing/freecad
COPY configs/manufacturing.example.yaml /opt/fuse-manufacturing/manufacturing.example.yaml

# Fail during the build if either the manufacturing backend or notebook server
# is inconsistent.
RUN python -m pip check \
    && python -c "import mapbox_earcut, plotly, trimesh, yaml; import fuse_manufacturing; print('FUSE manufacturing installation OK')" \
    && python -m jupyterlab --version

# Create a host-matched non-root identity only after privileged installation.
# Ubuntu 24.04 already reserves UID/GID 1000 for an `ubuntu` account, so reuse
# that identity when it exists instead of trying to create a duplicate GID.
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
    mkdir -p /home/fuse /workspace/data /workspace/notebooks /workspace/.cache; \
    chown -R "${USER_ID}:${GROUP_ID}" /workspace /home/fuse

USER fuse
WORKDIR /workspace

EXPOSE 8893

CMD ["/bin/bash"]
