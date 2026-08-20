# FUSE Repair Studio — Streamlit MVP

This app reads the existing Stage 1 and Stage 3 artifacts and provides four workflows:

1. inspect the fixed VGGT point cloud;
2. overlay the aligned/adapted Hunyuan–Kaolin hypothesis;
3. label unsupported connected components as `missing`, `false positive`, or `uncertain`;
4. add manual restoration points and create a guarded FreeCAD/verification hand-off.

The authority rule is unchanged: VGGT is measured evidence. The prior, selected fragments, and manually added points are hypotheses in the aligned VGGT coordinate frame.

## Expected paths

The app auto-discovers:

```text
/workspace/data/cleaned_geometry/global/broken_clean_normals.ply
/workspace/data/kaolin_outputs/alignment/runs/<run>/prior_adapted.glb
/workspace/data/kaolin_outputs/alignment/runs/<run>/prior_support_classification.ply
/workspace/data/kaolin_outputs/alignment/runs/<run>/alignment_report.json
```

Paths can also be overridden in the sidebar.

## Start inside `fuse-kaolin`

Rebuild the image once because Streamlit was added to the Dockerfile:

```bash
cd ~/Documents/FUSE/containers/kaolin
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
docker compose build kaolin
docker compose up -d kaolin
```

Start the app without stopping JupyterLab:

```bash
docker compose exec -d kaolin \
  python -m streamlit run /workspace/app/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501
```

Open <http://127.0.0.1:8501>.

To see Streamlit logs while debugging, omit `-d` from `docker compose exec`.

## Candidate review

Keep `Minimum candidate faces` equal to the notebook value (`40`) to preserve the same component IDs. Set the decision for each row:

- `uncertain`: keep for later inspection;
- `missing — send to verification`: merge into the yellow hand-off hypothesis;
- `false positive`: exclude and record as a negative human label.

The confirmation checkbox intentionally gates hand-off export. The bundle always contains a decision manifest and inspection formats. STL is included only when the selected fragment is watertight.

## Manual points

In the manual editor:

1. click a displayed VGGT point to set the red anchor;
2. enter `Δx`, `Δy`, and `Δz`;
3. add the next restoration point;
4. continue from the last point, edit exact coordinates in the table, or export CSV/PLY.

This first editor adds points relative to measured surface anchors. Arbitrary free-space coordinates can be entered in the absolute-coordinate panel.

## Stage 4 outputs

Saved hand-offs go to:

```text
/workspace/data/kaolin_outputs/alignment/runs/handoff_<UTC timestamp>/
```

The selected geometry is an exterior hypothesis. Before printing, Stage 4 still needs to construct the fracture mating surface, apply clearance, confirm physical scale, close the solid, and verify the fit in FreeCAD.
