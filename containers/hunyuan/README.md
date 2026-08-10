# FUSE Hunyuan3D-2.1 Container

This is the geometry-only 2nd container in the 3-container FUSE prototype:

```text
VGGT measured broken object
        +
Hunyuan3D-2.1 inferred intact prior (this project)
        ↓
Kaolin / geometry alignment, fragment construction and STL export
```

The image-to-shape stage is included. Hunyuan Paint, training, Blender, Gradio and API-server dependencies are intentionally omitted because FUSE needs an untextured mesh for alignment, not a production-rendered asset.

## Input

Put the intact reference image here:

```text
data/scenes/global/reference/pogo_complete.png
```

A clean image with the complete silhouette visible is best. A transparent background is ideal; the notebook can also run Hunyuan's background remover.

## Start

```bash
cd path/to/FUSE/containers/hunyuan

mkdir -p cache ../../notebooks ../../data
docker compose build hunyuan
docker compose up hunyuan
```

Open <http://localhost:8892> and run:

```text
notebooks/fuse-hunyuan.ipynb
```

The first run downloads the `tencent/Hunyuan3D-2.1` shape checkpoint into `cache/huggingface`. The checkpoint remains cached between container runs.

## Main output

After inspecting candidates and setting `SELECTED_SEED`, the notebook writes:

```text
data/hunyuan_outputs/runs/run_001/selected/intact_prior.glb
data/hunyuan_outputs/runs/run_001/selected/intact_prior.ply
data/hunyuan_outputs/runs/run_001/selected/intact_prior_preview.html
data/hunyuan_outputs/runs/run_001/selected/prior_manifest.json
```

`intact_prior.glb` is the handoff to the Kaolin/geometry container. It is an inferred, normalized, non-metric prior; Kaolin must estimate its scale, rotation and translation from the surviving common geometry.

## RTX 4080 Laptop settings

The notebook defaults to one seed and a 256 octree resolution for a quick first run. Once that works, use 384 for the final candidate. The official project reports about 10 GB VRAM for shape generation. If another process is occupying the GPU, stop it first.

Do not enable the texture stage on this 12 GB GPU: the official project reports about 21 GB VRAM for texture generation, and FUSE does not need it.
