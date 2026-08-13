# FUSE Repair Studio container

This CPU-only container runs the Streamlit review app independently from the
Kaolin alignment container. Both stages exchange artifacts through the shared
host `FUSE/data/` directory; no Docker network or simultaneous runtime is
required.

From `FUSE/containers/app`:

```bash
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
mkdir -p cache ../../app ../../data
docker compose up -d --build app
```

Open <http://127.0.0.1:8501>.

Useful checks:

```bash
docker compose ps
docker compose logs -f app
curl http://127.0.0.1:8501/_stcore/health
```

The app source is bind-mounted read-only from `FUSE/app/`. Stage 4 hand-offs
are written to `FUSE/data/04_verification/`.
