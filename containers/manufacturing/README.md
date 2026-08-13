# FUSE Manufacturing container 

FUSE Repair Studio is a standalone CPU-only Streamlit container. Kaolin is a single-purpose GPU/Jupyter container.

## Start

```bash

mkdir -p cache ../../notebooks ../../data
docker compose up -d --build --force-recreate manufacturing
```

Open <http://127.0.0.1:8893/lab/tree/notebooks/fuse-manufacturing.ipynb>.

## Standalone app

```bash
cd ~/Documents/FUSE/containers/app
mkdir -p cache ../../app ../../data
docker compose up -d --build --force-recreate app
```

Open <http://127.0.0.1:8501>.

The app reads Kaolin's completed artifacts through `FUSE/data/`; Kaolin may be
running or stopped. Check the app with:

```bash
docker compose logs --tail=100 app
curl http://127.0.0.1:8501/_stcore/health
```

The cleaned Kaolin service now exposes only JupyterLab on port 8891. Rebuilding
Kaolin is optional until its image or configuration otherwise needs changing.
