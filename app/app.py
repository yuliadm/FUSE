from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    save_fragment_edit,
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


def discover_fragment_edit_runs(data_root: Path, source_run: Path) -> list[Path]:
    runs_root = Path(data_root) / "kaolin_outputs" / "fragment_edits" / "runs"
    if not runs_root.exists():
        return []
    matches: list[Path] = []
    for candidate in runs_root.iterdir():
        manifest_path = candidate / "fragment_edit_manifest.json"
        if not candidate.is_dir() or not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if (
            manifest.get("schema") == "fuse.fragment-edit/v1"
            and manifest.get("source_alignment_run") == str(source_run)
        ):
            matches.append(candidate)
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)


def load_fragment_edit_manifest(edit_dir: Path) -> dict:
    manifest_path = Path(edit_dir) / "fragment_edit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "fuse.fragment-edit/v1":
        raise ValueError(f"Unsupported fragment-edit schema in {manifest_path}")
    return manifest


def manual_points_array() -> np.ndarray:
    points = st.session_state.get("manual_points", [])
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def set_manual_points(points: np.ndarray, *, refresh_editor: bool = True) -> None:
    st.session_state.manual_points = np.asarray(points, dtype=np.float64).reshape(-1, 3).tolist()
    if refresh_editor:
        st.session_state.manual_points_revision = (
            int(st.session_state.get("manual_points_revision", 0)) + 1
        )


def add_manual_point_callback() -> None:
    anchor = st.session_state.get("editor_anchor")
    if anchor is None:
        return
    anchor_array = np.asarray(anchor, dtype=np.float64).reshape(3)
    offset = np.asarray(
        [
            st.session_state.get("offset_x", 0.0),
            st.session_state.get("offset_y", 0.0),
            st.session_state.get("offset_z", 0.0),
        ],
        dtype=np.float64,
    )
    point = anchor_array + offset
    set_manual_points(np.vstack([manual_points_array(), point]))
    st.session_state.editor_anchor = point.tolist()
    st.session_state.offset_x = 0.0
    st.session_state.offset_y = 0.0
    st.session_state.offset_z = 0.0


def use_last_manual_point_callback() -> None:
    points = manual_points_array()
    if len(points):
        st.session_state.editor_anchor = points[-1].tolist()


def undo_manual_point_callback() -> None:
    points = manual_points_array()
    if len(points):
        set_manual_points(points[:-1])


def clear_manual_points_callback() -> None:
    set_manual_points(np.empty((0, 3), dtype=np.float64))
    st.session_state.editor_anchor = None
    st.session_state.last_plot_selection = None
    st.session_state.anchor_selector_revision = (
        int(st.session_state.get("anchor_selector_revision", 0)) + 1
    )


def reset_anchor_picker_callback() -> None:
    st.session_state.editor_anchor = None
    st.session_state.last_plot_selection = None
    st.session_state.anchor_selector_revision = (
        int(st.session_state.get("anchor_selector_revision", 0)) + 1
    )


def selected_projection_point(event) -> bool:
    try:
        points = event.selection.points
    except (AttributeError, KeyError, TypeError):
        return False
    if not points:
        return False
    point = points[-1]
    try:
        customdata = point["customdata"]
        coordinates = np.asarray(customdata[:3], dtype=np.float64)
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    if not np.isfinite(coordinates).all():
        return False
    signature = tuple(np.round(coordinates, 12))
    if (
        signature != st.session_state.get("last_plot_selection")
        or st.session_state.get("editor_anchor") is None
    ):
        st.session_state.last_plot_selection = signature
        st.session_state.editor_anchor = coordinates.tolist()
        st.session_state.offset_x = 0.0
        st.session_state.offset_y = 0.0
        st.session_state.offset_z = 0.0
        return True
    return False


def projection_selector_figure(
    measured,
    manual_points: np.ndarray,
    anchor: np.ndarray | None,
    projection: str,
    rotate_180: bool = True,
    maximum_measured: int = 70_000,
) -> go.Figure:
    axes = {
        "XY — view along Z": (0, 1, "x", "y"),
        "XZ — view along Y": (0, 2, "x", "z"),
        "YZ — view along X": (1, 2, "y", "z"),
    }
    horizontal, vertical, horizontal_name, vertical_name = axes[projection]

    count = len(measured.points)
    if count <= maximum_measured:
        indices = np.arange(count, dtype=np.int64)
    else:
        indices = np.sort(
            np.random.default_rng(17).choice(count, maximum_measured, replace=False)
        )
    measured_points = np.asarray(measured.points, dtype=np.float64)[indices]
    measured_customdata = np.column_stack(
        [
            measured_points,
            np.full(len(measured_points), "VGGT", dtype=object),
            indices.astype(object),
        ]
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=measured_points[:, horizontal],
            y=measured_points[:, vertical],
            mode="markers",
            name="VGGT measured",
            customdata=measured_customdata,
            marker={"size": 4, "color": "rgb(125,131,137)", "opacity": 0.58},
            hovertemplate=(
                "VGGT point %{customdata[4]}<br>"
                "x=%{customdata[0]:.7f}<br>"
                "y=%{customdata[1]:.7f}<br>"
                "z=%{customdata[2]:.7f}<extra></extra>"
            ),
        )
    )

    if len(manual_points):
        points = np.asarray(manual_points, dtype=np.float64).reshape(-1, 3)
        manual_customdata = np.column_stack(
            [
                points,
                np.full(len(points), "MANUAL", dtype=object),
                np.arange(len(points), dtype=object),
            ]
        )
        figure.add_trace(
            go.Scattergl(
                x=points[:, horizontal],
                y=points[:, vertical],
                mode="markers+lines",
                name="manual restoration points",
                customdata=manual_customdata,
                marker={"size": 9, "color": "rgb(255,77,157)", "symbol": "diamond"},
                line={"width": 2, "color": "rgb(255,77,157)"},
                hovertemplate=(
                    "manual point %{customdata[4]}<br>"
                    "x=%{customdata[0]:.7f}<br>"
                    "y=%{customdata[1]:.7f}<br>"
                    "z=%{customdata[2]:.7f}<extra></extra>"
                ),
            )
        )

    if anchor is not None:
        point = np.asarray(anchor, dtype=np.float64).reshape(3)
        figure.add_trace(
            go.Scattergl(
                x=[point[horizontal]],
                y=[point[vertical]],
                mode="markers",
                name="current anchor",
                customdata=[[point[0], point[1], point[2], "ANCHOR", 0]],
                marker={"size": 13, "color": "rgb(235,76,74)", "symbol": "x"},
                hovertemplate=(
                    "anchor<br>x=%{customdata[0]:.7f}<br>"
                    "y=%{customdata[1]:.7f}<br>"
                    "z=%{customdata[2]:.7f}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title={"text": f"{projection} — click a point to set the anchor", "x": 0.02},
        height=460,
        margin={"l": 20, "r": 20, "b": 20, "t": 55},
        clickmode="event+select",
        dragmode="zoom",
        hovermode="closest",
        uirevision=f"fuse-anchor-projection-{projection}-{rotate_180}",
        xaxis={
            "title": horizontal_name,
            "scaleanchor": "y",
            "scaleratio": 1,
            "autorange": "reversed" if rotate_180 else True,
        },
        yaxis={
            "title": vertical_name,
            "autorange": "reversed" if rotate_180 else True,
        },
        legend={"orientation": "h", "x": 0.0, "y": 1.08},
    )
    return figure


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
if "manual_points_revision" not in st.session_state:
    st.session_state.manual_points_revision = 0
if "anchor_selector_revision" not in st.session_state:
    st.session_state.anchor_selector_revision = 0
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
        "Rotate and inspect the object in the 3D view. Use the orthographic selector below "
        "to click a VGGT or manual point, then enter a local offset and add the next point."
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
        title="Manual repair editor — drag to orbit, scroll to zoom",
        maximum_measured=min(plot_max, 70_000),
        selectable=True,
    )
    editor_figure.update_scenes(
        dragmode="orbit",
        hovermode="closest",
        uirevision=f"manual-editor-camera-{run_key}",
    )
    st.plotly_chart(
        editor_figure,
        key=f"manual_editor_plot_{run_key}",
        width="stretch",
        theme=None,
        config={"scrollZoom": True, "displaylogo": False},
    )

    st.markdown("#### Choose an anchor")
    projection_col, orientation_col, reset_col = st.columns([3, 2, 1])
    with projection_col:
        projection = st.segmented_control(
            "Orthographic projection",
            options=["XY — view along Z", "XZ — view along Y", "YZ — view along X"],
            default="XY — view along Z",
            key=f"manual_anchor_projection_{run_key}",
        )
    with orientation_col:
        rotate_projection = st.toggle(
            "Turn projection upright (180°)",
            value=True,
            key=f"manual_anchor_rotate_{run_key}",
        )
    with reset_col:
        st.button(
            "Reset picker",
            width="stretch",
            key=f"reset_anchor_picker_{run_key}",
            on_click=reset_anchor_picker_callback,
        )
    if projection is None:
        projection = "XY — view along Z"
    selector_figure = projection_selector_figure(
        measured,
        manual_points,
        anchor_array,
        projection,
        rotate_180=rotate_projection,
        maximum_measured=min(plot_max, 70_000),
    )
    selector_revision = st.session_state.anchor_selector_revision
    selector_event = st.plotly_chart(
        selector_figure,
        key=(
            f"manual_anchor_selector_{projection}_{rotate_projection}_"
            f"{selector_revision}_{run_key}"
        ),
        on_select="rerun",
        selection_mode="points",
        width="stretch",
        theme=None,
        config={"scrollZoom": True, "displaylogo": False},
    )
    if selected_projection_point(selector_event):
        st.rerun()
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
            st.number_input("Δx", value=0.0, step=step, format="%.8f", key="offset_x")
        with dy_col:
            st.number_input("Δy", value=0.0, step=step, format="%.8f", key="offset_y")
        with dz_col:
            st.number_input("Δz", value=0.0, step=step, format="%.8f", key="offset_z")
        add_col, last_col, undo_col, clear_col = st.columns(4)
        add_col.button(
            "Add point at anchor + offset",
            type="primary",
            width="stretch",
            on_click=add_manual_point_callback,
        )
        last_col.button(
            "Use last manual point as anchor",
            width="stretch",
            disabled=not len(manual_points),
            on_click=use_last_manual_point_callback,
        )
        undo_col.button(
            "Undo last point",
            width="stretch",
            disabled=not len(manual_points),
            on_click=undo_manual_point_callback,
        )
        clear_col.button(
            "Clear manual points",
            width="stretch",
            disabled=not len(manual_points),
            on_click=clear_manual_points_callback,
        )

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
            key=(
                f"manual_point_table_{run_key}_"
                f"{st.session_state.manual_points_revision}"
            ),
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
            set_manual_points(cleaned, refresh_editor=False)

    manual_points = manual_points_array()
    st.metric("Manual points", f"{len(manual_points):,}")
    if len(manual_points):
        csv_bytes = manual_points_csv_bytes(manual_points)
        ply_bytes = point_cloud_ply_bytes(
            manual_points,
            np.tile(np.asarray([[255, 77, 157]], dtype=np.uint8), (len(manual_points), 1)),
        )
        dl_csv, dl_ply = st.columns(2)
        dl_csv.download_button(
            "Export manual points to browser · CSV",
            csv_bytes,
            "manual_restoration_points.csv",
            "text/csv",
            width="stretch",
        )
        dl_ply.download_button(
            "Export manual points to browser · PLY",
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
                "Export combined cloud to browser · PLY",
                point_cloud_ply_bytes(combined_points, combined_colors),
                "vggt_plus_manual_restoration_points.ply",
                "application/octet-stream",
                width="stretch",
            )

    st.divider()
    st.markdown("#### Save Stage 3 fragment draft")
    st.caption(
        "Persist the selected-fragment vertices and manual points as separate, provenance-aware "
        "artifacts before continuing to verification."
    )
    if selected_fragment is None and not len(manual_points):
        st.info("Select a missing component or add at least one manual point before saving a draft.")
    else:
        if st.button(
            "Save draft for Tab 4",
            type="primary",
            width="stretch",
            key=f"save_fragment_edit_{run_key}",
        ):
            output_dir = save_fragment_edit(
                data_root,
                run_dir,
                labels,
                selected_fragment,
                manual_points,
            )
            st.session_state.last_fragment_edit_dir = str(output_dir)
        last_fragment_edit = st.session_state.get("last_fragment_edit_dir")
        if last_fragment_edit:
            st.success(
                f"Stage 3 draft saved for Tab 4: `{last_fragment_edit}`. "
                "Open Tab 4 to verify the persisted hand-off."
            )

with tab_handoff:
    st.subheader("Stage 3 → Stage 4 verification hand-off")
    fragment_edit_runs = discover_fragment_edit_runs(data_root, run_dir)
    last_fragment_edit = st.session_state.get("last_fragment_edit_dir")
    if last_fragment_edit:
        last_path = Path(last_fragment_edit)
        if last_path.exists() and last_path not in fragment_edit_runs:
            try:
                last_manifest = load_fragment_edit_manifest(last_path)
            except (ValueError, json.JSONDecodeError, OSError):
                last_manifest = None
            if last_manifest and last_manifest.get("source_alignment_run") == str(run_dir):
                fragment_edit_runs.insert(0, last_path)

    if not fragment_edit_runs:
        st.info(
            "No persisted Stage 3 fragment draft is available for this alignment run. "
            "Return to Tab 3 and press **Save draft for Tab 4**. Browser export buttons do not "
            "advance the pipeline."
        )
    else:
        edit_names = [path.name for path in fragment_edit_runs]
        selected_edit_name = st.selectbox(
            "Persisted Stage 3 draft",
            edit_names,
            index=0,
            key=f"fragment_edit_handoff_{run_key}",
        )
        edit_dir = fragment_edit_runs[edit_names.index(selected_edit_name)]
        edit_manifest = load_fragment_edit_manifest(edit_dir)
        edit_mode = str(edit_manifest.get("mode", "unknown"))
        draft_selected_ids = [
            int(value) for value in edit_manifest.get("selected_component_ids", [])
        ]
        draft_labels = {item.component_id: "uncertain" for item in components}
        for component_id in draft_selected_ids:
            draft_labels[component_id] = "missing — send to verification"

        draft_selected_fragment = None
        if draft_selected_ids and classification_mesh is not None:
            draft_selected_fragment = merge_candidate_components(
                classification_mesh,
                components,
                draft_selected_ids,
            )
        elif int(edit_manifest.get("base_fragment", {}).get("point_count", 0)) > 0:
            draft_selected_fragment = existing_missing
        if (
            int(edit_manifest.get("base_fragment", {}).get("point_count", 0)) > 0
            and draft_selected_fragment is None
        ):
            st.error(
                "The persisted draft references an inferred/base fragment that cannot be "
                "resolved from the current alignment run. Restore the original candidate "
                "artifacts or save a new Stage 3 draft."
            )
            st.stop()

        manual_artifact = edit_manifest.get("artifacts", {}).get("manual_points")
        draft_manual_points = np.empty((0, 3), dtype=np.float64)
        if manual_artifact:
            manual_path = edit_dir / str(manual_artifact["path"])
            if manual_path.exists():
                draft_manual_points = load_cached_cloud(manual_path).points

        working_artifact = edit_manifest.get("artifacts", {}).get("working_fragment_points", {})
        working_path = edit_dir / str(working_artifact.get("path", ""))
        expected_working_point_count = int(
            edit_manifest.get("working_fragment", {}).get("point_count", 0)
        )
        base_point_count = int(edit_manifest.get("base_fragment", {}).get("point_count", 0))

        if not working_path.is_file():
            st.error(f"The persisted working fragment is missing: `{working_path}`")
            st.stop()
        draft_working_points = load_cached_cloud(working_path).points
        working_point_count = int(len(draft_working_points))
        if working_point_count != expected_working_point_count:
            st.error(
                "The persisted working-fragment point count does not match its manifest: "
                f"file={working_point_count:,}, manifest={expected_working_point_count:,}."
            )
            st.stop()

        st.success(f"Loaded persisted Stage 3 draft: `{edit_dir}`")
        mode_col, base_col, manual_col, working_col = st.columns(4)
        mode_col.metric("Hand-off mode", edit_mode.replace("_", " "))
        base_col.metric("Inferred/base points", f"{base_point_count:,}")
        manual_col.metric("Human-authored points", f"{len(draft_manual_points):,}")
        working_col.metric("Working points", f"{working_point_count:,}")

        st.plotly_chart(
            overlay_figure(
                measured,
                missing=draft_selected_fragment,
                manual_points=draft_manual_points,
                title="Persisted hand-off — yellow inferred, magenta human-authored",
                maximum_measured=min(plot_max, 70_000),
            ),
            key=f"persisted_handoff_preview_{selected_edit_name}_{run_key}",
            width="stretch",
            theme=None,
            config={"scrollZoom": True, "displaylogo": False},
        )

        st.markdown(
            "**This is a verification hand-off, not a print command.** The magenta point cloud "
            "is a valid human-authored Stage 3 hypothesis, but it is not yet a printable surface. "
            "Stage 4 must construct or refine the surface, build the mating interface, add "
            "clearance, confirm scale, close the solid, and verify fit."
        )
        if draft_selected_fragment is not None and not draft_selected_fragment.is_watertight:
            st.warning(
                "The yellow exterior is open, so STL is intentionally withheld. OBJ, PLY, and "
                "GLB remain available for verification and completion."
            )

        confirmation_labels = {
            "manual_only": (
                "I confirm that the magenta manual point cloud is the intended missing-fragment "
                "hypothesis for Stage 4 verification."
            ),
            "selected_only": (
                "I confirm that the yellow selected components are genuinely absent—not merely "
                "unobserved or mismatched."
            ),
            "selected_plus_manual": (
                "I confirm that the yellow selected components and magenta manual point cloud "
                "together form the intended missing-fragment hypothesis for Stage 4 verification."
            ),
        }
        confirmation = st.checkbox(
            confirmation_labels.get(
                edit_mode,
                "I confirm this persisted Stage 3 draft for Stage 4 verification.",
            ),
            value=False,
            key=f"confirm_fragment_edit_{selected_edit_name}",
        )
        if confirmation:
            zip_bytes = build_handoff_zip(
                run_dir,
                draft_labels,
                draft_selected_fragment,
                draft_manual_points,
            )
            export_col, save_col = st.columns(2)
            export_col.download_button(
                "Export verification ZIP to browser",
                zip_bytes,
                f"fuse_{run_dir.name}_freecad_handoff.zip",
                "application/zip",
                width="stretch",
            )
            if save_col.button(
                "Commit persistent Stage 4 run",
                type="primary",
                width="stretch",
                key=f"save_stage_4_{selected_edit_name}",
            ):
                output_dir = save_handoff(
                    data_root,
                    run_dir,
                    draft_labels,
                    draft_selected_fragment,
                    draft_manual_points,
                )
                st.success(f"Persistent Stage 4 run saved: `{output_dir}`")
        else:
            st.caption("Confirmation is required before Stage 4 export or commit is enabled.")

with st.sidebar:
    st.divider()
    st.caption("All coordinates remain in VGGT units. FreeCAD physical scale is a Stage 4 verification item.")
