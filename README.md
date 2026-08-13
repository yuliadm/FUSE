# FUSE  (Fragment Understanding with Surface Ergodics)

<img src="assets/fuse-banner.png" alt="Logo" width="1000">

**FUSE** is an AI framework that reconstructs missing fragments of broken cultural or decorative objects from visual evidence, generates a 3D-printable replacement mesh, and plans adaptive surface finishing
paths after assembly. It is designed for museums, restoration studios, heritage institutions, artists working with broken or incomplete cultural and decorative objects (reliefs, design artifacts, sculptural fragments,
etc.), and individuals who are reluctant to throw away memorable items whenever they get damaged.

**FUSE** connects directly to the creative and cultural industries by turning restoration into a digitally assisted, sustainable workflow. Instead of discarding damaged objects or relying on manual reconstruction,
one can scan the broken area, provide visual reference of the original object (one reference image + a 3D scan of the broken object), and generate a printable replacement fragment. After the printed piece has
been inserted, the glue line still needs careful finishing. Our HEDAC-based ergodic controller computes a robot-arm path that concentrates polishing along the seam, while still covering the surrounding surface so
the repair blends naturally into the object.

<img src="assets/fuse.png" alt="Logo" width="1000">


**FUSE** is a mix of software components based on deep learning methods (point cloud extraction from video images, 3D missing shape reconstruction, mesh construction from point clouds) and robot learning
techniques.




The repo includes the experimental data and 3 working containers + 1 container under dev (folder `containers`):
- fuse-vggt
- fuse-hunyuan
- fuse-kaolin
- manufacturing


At the moment, there is no orchestrator yet, so each container has to be run manually. 

## Prerequisites:
- Linux with an NVIDIA GPU;
- a working NVIDIA driver;
- Docker Engine;
- the Docker Compose plugin;
- NVIDIA Container Toolkit.



## Instructions:
1. Verify the host setup:
```bash
docker --version
docker compose version
nvidia-smi
```

2. Verify that Docker can access the GPU:
```bash
docker run --rm --gpus all \
  nvidia/cuda:12.4.1-base-ubuntu22.04 \
  nvidia-smi    
```

3. Build/Start the container (e.g. hunyuan)

```bash
cd path/to/FUSE/containers/hunyuan

mkdir -p cache ../../notebooks ../../data
docker compose build hunyuan
docker compose up hunyuan
```

4. To access the notebook, open <http://localhost:8892> in the browser and run cells in
`notebooks/fuse-hunyuan.ipynb`

**NOTE**: each container uses a separate local host port (to enable parallel processing) 
- vggt: 8890
- hunyuan: 8892
- kaolin: 8891
- manufacturing: 8893


## Container description


#### Container 1 (fuse-vggt). 

The Dockerfile and compose.yaml: `containers/vggt`.
**Input data** (video of the broken object): `data/scenes/global/raw_video/video.mp4`
The corresponding Jupyter notebook: `notebooks/fuse-vggt.ipynb`

**Main tasks** performed: 
 - frames extraction from the video of the broken object
 - background removal
 - VGGT pointcloud restoration
 - outlier removal  

**Outputs**: `data/vggt_outputs/global/broken_clean_normals.ply` (pointcloud of the broken object)

Output vizualization: FUSE/data/vggt_outputs/global/broken_clean.html


<img src="assets/frame_001.jpg" alt="Broken object" width="500">  

<img src="assets/vggt_out.png" alt="VGGT output" width="500" height="450">


#### Container 2 (fuse-hunyuan).

The Dockerfile and compose.yaml: `containers/vggt`.
**Input data** (reference image of the intact object): `data/scenes/global/reference/pogo_complete.png`
The corresponding Jupyter notebook: `notebooks/fuse-hunyuan.ipynb`

**Main tasks**:
 - optional background removal in the reference image
 - 3D shape restoration from a single image of an object (using Hunyuan3D 2.1)
 - best candidate selection


**Outputs**: `data/hunyuan_outputs/runs/run_001/selected/intact_prior.ply` (complete object prior pointcloud), `data/hunyuan_outputs/runs/run_001/selected/intact_prior.glb`

Output vizualization: `data/hunyuan_outputs/runs/run_001/selected/intact_prior_preview.html`

<img src="assets/hunyuan_out.png" alt="Hunyuan output" width="500" height="400">


#### Container 3 (fuse-kaolin).

The Dockerfile and compose.yaml: `containers/kaolin`.
**Input data** (outputs of Container 1 and Container 2): `data/vggt_outputs/global/broken_clean_normals.ply`, `FUSE/data/hunyuan_outputs/runs/run_001/selected/intact_prior.ply`
The corresponding Jupyter notebook: `notebooks/fuse-kaolin.ipynb`

**Main tasks**:
 - VGGT shape (broken) and Hunyuan shape (complete) preliminary alignment using PCA and post-Sim(3) - handle rotation, translation, scale
 - differentiable fitting, silhouette verification and fracture-interface fitting using Kaolin
 - computation of shape mismatch to obtain candidate missing fragments


**Outputs**: `data/kaolin_outputs/alignment/runs/alignment_20260809T214710Z/prior_support_classification.ply`

Output vizualization: `data/kaolin_outputs/alignment/runs/alignment_20260809T214710Z/04_support_classification.html`, `data/kaolin_outputs/alignment/runs/alignment_20260809T214710Z/05_missing_candidates.html`

<img src="assets/kaolin_out1.png" alt="Kaolin output supports" width="500" height="450"> 

<img src="assets/kaolin_out2.png" alt="Kaolin output fragment candidates" width="500" height="450">




