# FUSE Stage 0 — Synthetic intact-reference hypotheses

This module uses `black-forest-labs/FLUX.2-klein-4B` to propose intact reference
images from a photograph of a damaged object. Its output is a **synthetic shape
hypothesis**, not measured evidence and not ground truth.

The notebook generates multiple seeded candidates. A user-defined mask limits
which pixels may enter the published result: outside the binary mask, the
original photograph is copied back exactly. The selected reference can then be
published explicitly to the image path consumed by the Hunyuan notebook.

## Project placement

Merge this bundle into the existing FUSE project so the files appear at:

```text
FUSE/
├── containers/reference/Dockerfile
├── containers/reference/compose.yaml
├── notebooks/fuse-reference-generation.ipynb
└── data/reference_generation/
    ├── inputs/pogo_incomplete.png
    └── previous_synthetic/pogo_complete.png
```

The second image is provided only for visual comparison and is never treated as
ground truth or passed to the model by default.

## Build and start

```bash
cd ~/Documents/FUSE/containers/reference

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

mkdir -p cache

docker compose build reference
docker compose up -d reference
docker compose logs --tail=100 reference
```

Open:

```text
http://127.0.0.1:8895/lab/tree/notebooks/fuse-reference-generation.ipynb
```

The first model run downloads the FLUX.2 Klein weights into
`containers/reference/cache`; later runs reuse them.

## GPU and RAM policy

The notebook defaults to model-level CPU offload and processes one candidate at
a time. This is intended for one RTX 4080. If a 12 GB laptop RTX 4080 still
raises CUDA OOM, change `OFFLOAD_MODE` to `"sequential"` in the configuration
cell and rerun from the pipeline-loading cell. Sequential offload is slower but
uses less VRAM. Host RAM of 32 GB or more is recommended.

## Outputs

Each run is written under:

```text
data/reference_outputs/runs/reference_<UTC timestamp>/
```

It contains the input snapshot, binary and feathered masks, raw model outputs,
pixel-constrained candidates, an ensemble-variance visualization, a selected
candidate, and `reference_manifest.json`.

Publishing to Hunyuan is disabled by default. After inspection, set
`PUBLISH_TO_HUNYUAN = True`; the notebook then copies the chosen image to:

```text
data/scenes/global/reference/pogo_complete.png
```

An existing file is backed up before replacement.
