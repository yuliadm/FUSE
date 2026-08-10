# FUSE VGGT Container 

Tasks:
- frame extraction
- image preprocessing
    VGGT inference
    point cloud cleanup
    ROI/fracture diagnostics
    Open3D visualization of the restored point cloud
    


Contents:
This container provides the VGGT reconstruction model, FUSE point-cloud utilities, JupyterLab, and the dependencies required by VGGT’s COLMAP bundle-adjustment demo.


Prerequisites:
Linux with an NVIDIA GPU;
a working NVIDIA driver;
Docker Engine;
the Docker Compose plugin;
NVIDIA Container Toolkit.

Verify the host setup:
```
docker --version
docker compose version
nvidia-smi
```

Verify that Docker can access the GPU:
```
docker run --rm --gpus all \
  nvidia/cuda:12.4.1-base-ubuntu22.04 \
  nvidia-smi    
```


## Start

```bash
cd path/to/FUSE/containers/vggt


mkdir -p cache ../../notebooks ../../data
docker compose build vggt
docker compose up vggt
```

Open <http://localhost:8890> and run:
```
notebooks/fuse-vggt.ipynb
```


