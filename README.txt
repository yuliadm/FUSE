Build from project root, but point to the correct docker file: build -t fuse-vggt -f containers/vggt/Dockerfile .
Run the container: docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace \
  -v "$PWD/data/cache":/workspace/.cache \
  -w /workspace \
  fuse-vggt bash
Check: pwd
ls -lah
ls -lah scripts
Run the VGGT model script: 
python3 scripts/vggt_model.py \
  --scene global \
  --image-dir data/scenes/global/raw_images \
  --use-rembg \
  --max-frames 12 \
  --frame-step 4 \
  --conf-quantile 0.4




The 4 containers
- vggt
- photogrammetry
- kaolin
- tools
communicate only through files in the data/. folder to ensure modularity.

1. vggt container

Purpose: raw images / video → VGGT cameras, depth, point maps, point cloud
Output:

data/vggt_outputs/global_cloud.ply
data/vggt_outputs/global_cameras.json
data/vggt_outputs/depth_maps/

This container does not know anything about Kaolin.


2. photogrammetry container

Purpose: close-up fracture photos → local high-precision fracture cloud / mesh
Tools could be: COLMAP + OpenMVS or Meshroom/AliceVision
Output:

data/photogrammetry_outputs/fracture_local_cloud.ply
data/photogrammetry_outputs/fracture_local_mesh.ply

This is the precision backup if VGGT is too soft near the fracture.


3. tools container

Purpose: Open3D / trimesh / pymeshlab / scipy / opencv cleanup
Tasks:

outlier removal
normal estimation
local registration
mesh repair
boundary extraction
ROI cropping
visual diagnostics

Output:

data/cleaned_geometry/broken_global_clean.ply
data/cleaned_geometry/fracture_local_clean.ply
data/cleaned_geometry/broken_fused_mesh.ply
data/cleaned_geometry/fracture_boundary_curve.ply
data/cleaned_geometry/fracture_boundary_points.npy


4. kaolin container

Purpose:

differentiable rendering
silhouette alignment
boundary constraint optimization
loss experiments

Output:

data/kaolin_outputs/alignment_params.json
data/kaolin_outputs/rendered_silhouette.png
data/kaolin_outputs/aligned_broken_mesh.ply
data/kaolin_outputs/boundary_fit_report.json

This container receives cleaned geometry, not raw messy scans.


5. lightweight orchestrator

This can just be host-level:

Makefile
docker compose
pipeline.yaml

I would not build a heavy “master container” yet. A master container is useful later, but for now a Makefile is more transparent.

Example:

frames:
	docker compose run --rm tools python scripts/00_extract_frames.py

vggt:
	docker compose run --rm vggt python scripts/02_run_vggt.py

clean:
	docker compose run --rm tools python scripts/03_clean_pointcloud.py

fracture:
	docker compose run --rm tools python scripts/05_extract_fracture_boundary.py

kaolin-align:
	docker compose run --rm kaolin python scripts/06_kaolin_silhouette_alignment.py

This is easier to debug than a fully automated master script.


