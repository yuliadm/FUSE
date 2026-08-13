from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from fuse_ui.geometry import (
    build_candidate_components,
    build_handoff_zip,
    discover_alignment_runs,
    discover_inputs,
    load_mesh,
    load_point_cloud,
    manual_points_csv_bytes,
    merge_candidate_components,
    point_cloud_ply_bytes,
    save_handoff,
)
from fuse_ui.plotting import candidate_figure, overlay_figure


st.set_page_config(
    page_title="FUSE Repair Studio",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.25rem; padding-bottom: 3rem;}
      [data-testid="stMetricValue"] {font-size: 1.25rem;}
      .fuse-rule {border-left: 4px solid #ffd728; padding: .4rem .8rem; background: #fff9d7;}
      .small-note {color: #60656b; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


REVIEW_OPTIONS = [
    "uncertain",
    "false positive",
]


@st.cache_resource(show_spinner="Loading VGGT cloud …")
def cached_cloud(path: str, modified_ns: int):
    del modified_ns
    return load_point_cloud(path)


@st.cache_resource(show_spinner="Loading mesh …")
def cached_mesh(path: str, modified_ns: int):
    del modified_ns
    return load_mesh(path)


@st.cache_resource(show_spinner="Reconstructing candidate components …")
def cached_candidates(path: str, modified_ns: int, min_faces: int):
    del modified_ns
    mesh = load_mesh(path)
    return mesh, build_candidate_components(mesh, min_faces=min_faces)


def load_cached_cloud(path: Path):
    return cached_cloud(str(path), path.stat().st_mtime_ns)


def load_cached_mesh(path: Path):
    return cached_mesh(str(path), path.stat().st_mtime_ns)


def path_input(label: str, value: Path | None, key: str) -> Path | None:
    raw = st.text_input(label, value="" if value is None else str(value), key=key).strip()
    if not raw:
        return None
    return Path(raw)


def manual_points_array() -> np.ndarray:
    points = st.session_state.get("manual_points", [])
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def set_manual_points(points: np.ndarray) -> None:
    st.session_state.manual_points = np.asarray(points, dtype=np.float64).reshape(-1, 3).tolist()


def selected_plot_point(event) -> np.ndarray | None:
    try:
        points = event.selection.points
    except (AttributeError, KeyError, TypeError):
        return None
    if not points:
        return None
    point = points[-1]
    try:
        coordinates = np.asarray([point["x"], point["y"], point["z"]], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(coordinates).all():
        return None
    signature = tuple(np.round(coordinates, 12))
    if signature != st.session_state.get("last_plot_selection"):
        st.session_state.last_plot_selection = signature
        st.session_state.editor_anchor = coordinates.tolist()
        st.session_state.offset_x = 0.0
        st.session_state.offset_y = 0.0
        st.session_state.offset_z = 0.0
    return coordinates


def candidate_table(
    components,
    run_key: str,
) -> tuple[pd.DataFrame, dict[int, str], list[int]]:
    rows = [
        {
            "component_id": int(item.component_id),
            "send_to_freecad": False,
            "review": "uncertain",
            "faces": len(item.faces),
            "area": item.area,
            "centroid_x": item.centroid[0],
            "centroid_y": item.centroid[1],
            "centroid_z": item.centroid[2],
        }
        for item in components
    ]

    frame = pd.DataFrame(rows)

    # v2 avoids reusing incompatible state from the old table.
    editor_key = f"candidate_selection_v2_{run_key}"

    edited = st.data_editor(
        frame,
        key=editor_key,
        width="stretch",
        hide_index=True,
        disabled=[
            "component_id",
            "faces",
            "area",
            "centroid_x",
            "centroid_y",
            "centroid_z",
        ],
        column_config={
            "component_id": st.column_config.NumberColumn(
                "ID",
                format="%d",
            ),
            "send_to_freecad": st.column_config.CheckboxColumn(
                "Include in FreeCAD hand-off",
                help="Select all fragments that belong to the missing piece.",
                default=False,
            ),
            "review": st.column_config.SelectboxColumn(
                "Review",
                options=REVIEW_OPTIONS,
                required=True,
            ),
            "faces": st.column_config.NumberColumn(
                "Faces",
                format="%d",
            ),
            "area": st.column_config.NumberColumn(
                "Area",
                format="%.6g",
            ),
            "centroid_x": st.column_config.NumberColumn(
                "cx",
                format="%.5f",
            ),
            "centroid_y": st.column_config.NumberColumn(
                "cy",
                format="%.5f",
            ),
            "centroid_z": st.column_config.NumberColumn(
                "cz",
                format="%.5f",
            ),
        },
    )

    labels: dict[int, str] = {}
    selected_ids: list[int] = []

    for row in edited.itertuples(index=False):
        component_id = int(row.component_id)

        if bool(row.send_to_freecad):
            labels[component_id] = "missing — send to verification"
            selected_ids.append(component_id)
        else:
            labels[component_id] = str(row.review)

    return edited, labels, selected_ids


def resolve_path_or_stop(path: Path | None, label: str) -> Path:
    if path is None:
        st.error(f"No {label} was discovered. Set its path in the sidebar.")
        st.stop()
    if not path.exists():
        st.error(f"{label} does not exist: `{path}`")
        st.stop()
    return path


if "manual_points" not in st.session_state:
    st.session_state.manual_points = []
if "editor_anchor" not in st.session_state:
    st.session_state.editor_anchor = None

st.title("FUSE Repair Studio")
st.markdown(
    '<div class="fuse-rule"><b>Evidence rule:</b> VGGT stays fixed. '
    "Hunyuan/Kaolin, candidate fragments, and manual points are hypotheses in the aligned VGGT frame.</div>",
    unsafe_allow_html=True,
)

default_root = Path(os.environ.get("FUSE_ROOT", "/workspace"))
with st.sidebar:
    st.header("Project data")
    fuse_root = Path(st.text_input("FUSE root", str(default_root))).expanduser()
    data_root = fuse_root / "data"
    runs = discover_alignment_runs(data_root)
    if not runs:
        st.error(f"No Stage 3 runs found under `{data_root / 'kaolin_outputs/alignment/runs'}`")
        st.stop()
    run_names = [path.name for path in runs]
    selected_run_name = st.selectbox("Alignment run", run_names, index=0)
    run_dir = runs[run_names.index(selected_run_name)]
    sidebar_run_key = str(abs(hash(str(run_dir))))
    discovered = discover_inputs(fuse_root, run_dir)

    with st.expander("Resolved artifact paths", expanded=False):
        measured_path = path_input(
            "VGGT cloud", discovered["measured"], f"measured_path_{sidebar_run_key}"
        )
        prior_path = path_input(
            "Aligned/adapted prior", discovered["prior"], f"prior_path_{sidebar_run_key}"
        )
        classification_path = path_input(
            "Support classification PLY",
            discovered["classification"],
            f"classification_path_{sidebar_run_key}",
        )
        missing_path = path_input(
            "Existing missing-piece export (optional)",
            discovered["missing"],
            f"missing_path_{sidebar_run_key}",
        )
    plot_max = st.slider("VGGT display points", 10_000, 120_000, 70_000, 5_000)
    use_source_colors = st.toggle("Use VGGT source colours", value=False)
    min_component_faces = st.number_input(
        "Minimum candidate faces",
        min_value=1,
        max_value=10_000,
        value=40,
        step=10,
        help="Keep this at the notebook value (40) if you want the same component IDs.",
    )

measured_path = resolve_path_or_stop(measured_path, "VGGT cloud")
prior_path = resolve_path_or_stop(prior_path, "aligned/adapted prior")
measured = load_cached_cloud(measured_path)
prior_mesh = load_cached_mesh(prior_path)

classification_mesh = None
components = []
if classification_path is not None and classification_path.exists():
    classification_mesh, components = cached_candidates(
        str(classification_path),
        classification_path.stat().st_mtime_ns,
        int(min_component_faces),
    )

existing_missing = None
if missing_path is not None and missing_path.exists():
    existing_missing = load_cached_mesh(missing_path)

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("VGGT points", f"{len(measured.points):,}")
metric_b.metric("Prior faces", f"{len(prior_mesh.faces):,}")
metric_c.metric("Candidate components", f"{len(components):,}")
metric_d.metric("VGGT diagonal", f"{measured.diagonal:.6g}")

tab_geometry, tab_review, tab_manual, tab_handoff = st.tabs(
    ["1 · Geometry", "2 · Candidate review", "3 · Manual points", "4 · FreeCAD hand-off"]
)

run_key = str(abs(hash(str(run_dir))))
labels: dict[int, str] = {item.component_id: "uncertain" for item in components}
selected_fragment = existing_missing

with tab_geometry:
    measured_tab, prior_tab, missing_tab = st.tabs(
        ["VGGT point cloud", "Hunyuan/Kaolin hypothesis", "Missing-fragment hypothesis"]
    )
    with measured_tab:
        st.plotly_chart(
            overlay_figure(
                measured,
                title="VGGT measured broken geometry",
                maximum_measured=plot_max,
                use_source_colors=use_source_colors,
            ),
            width="stretch",
            theme=None,
            config={"scrollZoom": True, "displaylogo": False},
        )
    with prior_tab:
        st.plotly_chart(
            overlay_figure(
                measured,
                prior=prior_mesh,
                title="Aligned/adapted Hunyuan hypothesis over fixed VGGT geometry",
                maximum_measured=plot_max,
                use_source_colors=use_source_colors,
            ),
            width="stretch",
            theme=None,
            config={"scrollZoom": True, "displaylogo": False},
        )
    with missing_tab:
        if existing_missing is None:
            st.info(
                "No missing-piece export exists yet. Label candidate components in the next tab; "
                "the yellow preview will then become the current hypothesis."
            )
        else:
            st.plotly_chart(
                overlay_figure(
                    measured,
                    missing=existing_missing,
                    title="FUSE completion hypothesis — grey measured, yellow inferred",
                    maximum_measured=plot_max,
                ),
                width="stretch",
                theme=None,
                config={"scrollZoom": True, "displaylogo": False},
            )

with tab_review:
    st.subheader("Human review of missing-fragment candidates")
    st.caption(
    "Use the component IDs shown in the 3D plot. Check every component "
    "that belongs to the missing fragment. Selected components become "
    "yellow and are merged for the FreeCAD hand-off."
    )
    if classification_mesh is None:
        st.warning("A support-classification PLY is required for candidate review.")
    elif not components:
        st.warning("No candidate components survived the current minimum-face threshold.")
    else:
        _, labels, selected_ids = candidate_table(
            components,
            run_key,
)
        selected_fragment = merge_candidate_components(
            classification_mesh,
            components,
            selected_ids,
        )
        st.plotly_chart(
            candidate_figure(
                measured,
                classification_mesh,
                components,
                labels,
                maximum_measured=min(plot_max, 60_000),
            ),
            width="stretch",
            theme=None,
            config={"scrollZoom": True, "displaylogo": False},
        )
        if selected_fragment is None:
            st.info("No component is currently labelled as missing.")
        else:
            left, middle, right = st.columns(3)
            left.metric("Selected IDs", ", ".join(map(str, selected_ids)))
            middle.metric("Selected faces", f"{len(selected_fragment.faces):,}")
            right.metric("Watertight", "yes" if selected_fragment.is_watertight else "no")
            st.plotly_chart(
                overlay_figure(
                    measured,
                    missing=selected_fragment,
                    title="Human-selected fragment hypothesis",
                    maximum_measured=plot_max,
                ),
                width="stretch",
                theme=None,
                config={"scrollZoom": True, "displaylogo": False},
            )

with tab_manual:
    st.subheader("Manual restoration-point editor")
    st.caption(
        "Click a displayed VGGT or manual point to use it as the anchor, then enter a local "
        "offset and add the next point. Direct coordinate entry remains available below."
    )
    manual_points = manual_points_array()
    anchor = st.session_state.get("editor_anchor")
    anchor_array = None if anchor is None else np.asarray(anchor, dtype=np.float64)
    editor_figure = overlay_figure(
        measured,
        prior=prior_mesh if st.toggle("Show prior as a guide", value=False) else None,
        missing=selected_fragment if st.toggle("Show selected fragment", value=True) else None,
        manual_points=manual_points,
        anchor=anchor_array,
        title="Manual repair editor — click a point to set the red anchor",
        maximum_measured=min(plot_max, 70_000),
        selectable=True,
    )
    event = st.plotly_chart(
        editor_figure,
        key=f"manual_editor_plot_{run_key}",
        on_select="rerun",
        selection_mode="points",
        width="stretch",
        theme=None,
        config={"scrollZoom": True, "displaylogo": False},
    )
    selected_plot_point(event)
    anchor = st.session_state.get("editor_anchor")
    anchor_array = None if anchor is None else np.asarray(anchor, dtype=np.float64)

    step = max(measured.diagonal / 500.0, 1e-7)
    if anchor_array is None:
        st.info("Select a cloud point above, or set an anchor with the absolute coordinates below.")
    else:
        st.code(
            f"anchor = [{anchor_array[0]:.8g}, {anchor_array[1]:.8g}, {anchor_array[2]:.8g}]",
            language="text",
        )
        dx_col, dy_col, dz_col = st.columns(3)
        with dx_col:
            dx = st.number_input("Δx", value=0.0, step=step, format="%.8f", key="offset_x")
        with dy_col:
            dy = st.number_input("Δy", value=0.0, step=step, format="%.8f", key="offset_y")
        with dz_col:
            dz = st.number_input("Δz", value=0.0, step=step, format="%.8f", key="offset_z")
        add_col, last_col, undo_col = st.columns(3)
        if add_col.button("Add point at anchor + offset", type="primary", width="stretch"):
            point = anchor_array + np.asarray([dx, dy, dz], dtype=np.float64)
            points = np.vstack([manual_points_array(), point])
            set_manual_points(points)
            st.session_state.editor_anchor = point.tolist()
            st.session_state.offset_x = 0.0
            st.session_state.offset_y = 0.0
            st.session_state.offset_z = 0.0
            st.rerun()
        if last_col.button("Use last manual point as anchor", width="stretch", disabled=not len(manual_points)):
            st.session_state.editor_anchor = manual_points[-1].tolist()
            st.rerun()
        if undo_col.button("Undo last point", width="stretch", disabled=not len(manual_points)):
            set_manual_points(manual_points[:-1])
            st.rerun()

    with st.expander("Absolute coordinates and table editor", expanded=anchor_array is None):
        base = np.zeros(3) if anchor_array is None else anchor_array
        ax, ay, az = st.columns(3)
        absolute_x = ax.number_input("x", value=float(base[0]), format="%.9f", key="absolute_x")
        absolute_y = ay.number_input("y", value=float(base[1]), format="%.9f", key="absolute_y")
        absolute_z = az.number_input("z", value=float(base[2]), format="%.9f", key="absolute_z")
        if st.button("Set absolute anchor"):
            st.session_state.editor_anchor = [absolute_x, absolute_y, absolute_z]
            st.session_state.last_plot_selection = None
            st.rerun()

        manual_frame = pd.DataFrame(manual_points_array(), columns=["x", "y", "z"])
        edited_manual = st.data_editor(
            manual_frame,
            key=f"manual_point_table_{run_key}",
            num_rows="dynamic",
            hide_index=False,
            width="stretch",
            column_config={
                "x": st.column_config.NumberColumn(format="%.9f", required=True),
                "y": st.column_config.NumberColumn(format="%.9f", required=True),
                "z": st.column_config.NumberColumn(format="%.9f", required=True),
            },
        )
        cleaned = edited_manual.apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=np.float64)
        current = manual_points_array()
        if cleaned.shape != current.shape or (
            cleaned.size and not np.allclose(cleaned, current, rtol=0.0, atol=1e-12)
        ):
            set_manual_points(cleaned)

    manual_points = manual_points_array()
    if len(manual_points):
        csv_bytes = manual_points_csv_bytes(manual_points)
        ply_bytes = point_cloud_ply_bytes(
            manual_points,
            np.tile(np.asarray([[255, 77, 157]], dtype=np.uint8), (len(manual_points), 1)),
        )
        dl_csv, dl_ply = st.columns(2)
        dl_csv.download_button(
            "Download manual points · CSV",
            csv_bytes,
            "manual_restoration_points.csv",
            "text/csv",
            width="stretch",
        )
        dl_ply.download_button(
            "Download manual points · PLY",
            ply_bytes,
            "manual_restoration_points.ply",
            "application/octet-stream",
            width="stretch",
        )
        if st.checkbox(
            "Prepare a combined VGGT + manual-points PLY",
            value=False,
            help="The measured points remain unchanged; manual points are appended in magenta.",
        ):
            combined_points = np.vstack([measured.points, manual_points])
            combined_colors = np.vstack(
                [
                    measured.colors,
                    np.tile(
                        np.asarray([[255, 77, 157]], dtype=np.float64) / 255.0,
                        (len(manual_points), 1),
                    ),
                ]
            )
            st.download_button(
                "Download combined cloud · PLY",
                point_cloud_ply_bytes(combined_points, combined_colors),
                "vggt_plus_manual_restoration_points.ply",
                "application/octet-stream",
                width="stretch",
            )

with tab_handoff:
    st.subheader("FreeCAD and verification hand-off")
    manual_points = manual_points_array()
    selected_ids = sorted(
        component_id
        for component_id, label in labels.items()
        if label == "missing — send to verification"
    )
    if selected_fragment is None and not len(manual_points):
        st.info("Select at least one missing component or add manual restoration points first.")
    else:
        st.markdown(
            "**This is a verification hand-off, not a print command.** The selected mesh is an "
            "exterior hypothesis. The next stage must construct the mating surface from the "
            "fracture, add clearance, confirm physical scale, close the solid, and verify fit."
        )
        if selected_fragment is not None:
            health_a, health_b, health_c = st.columns(3)
            health_a.metric("Components", ", ".join(map(str, selected_ids)) or "existing export")
            health_b.metric("Faces", f"{len(selected_fragment.faces):,}")
            health_c.metric("STL gate", "open" if selected_fragment.is_watertight else "blocked")
            if not selected_fragment.is_watertight:
                st.warning(
                    "The selected exterior is open, so STL is intentionally withheld. OBJ, PLY, "
                    "and GLB remain available for FreeCAD inspection and completion."
                )

        confirmation = st.checkbox(
            "I confirm that the yellow components are genuinely absent—not merely unobserved or mismatched.",
            value=False,
        )
        if confirmation:
            zip_bytes = build_handoff_zip(
                run_dir,
                labels,
                selected_fragment,
                manual_points,
            )
            download_col, save_col = st.columns(2)
            download_col.download_button(
                "Download verification bundle",
                zip_bytes,
                f"fuse_{run_dir.name}_freecad_handoff.zip",
                "application/zip",
                type="primary",
                width="stretch",
            )
            if save_col.button("Save as Stage 4 run", width="stretch"):
                output_dir = save_handoff(
                    data_root,
                    run_dir,
                    labels,
                    selected_fragment,
                    manual_points,
                )
                st.success(f"Saved: `{output_dir}`")
        else:
            st.caption("Confirmation is required before the hand-off bundle is enabled.")

with st.sidebar:
    st.divider()
    st.caption("All coordinates remain in VGGT units. FreeCAD physical scale is a Stage 4 verification item.")
