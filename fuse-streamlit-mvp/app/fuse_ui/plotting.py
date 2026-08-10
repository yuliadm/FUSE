from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go
import trimesh

from .geometry import CandidateComponent, GeometryBundle


VGGT_GREY = "rgb(125,131,137)"
PRIOR_CYAN = "rgb(22,184,212)"
MISSING_YELLOW = "rgb(255,215,40)"
MANUAL_MAGENTA = "rgb(255,77,157)"
ANCHOR_RED = "rgb(235,76,74)"
CANDIDATE_PALETTE = [
    "#ffd92f",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#e5c494",
    "#b3b3b3",
    "#1f78b4",
    "#33a02c",
    "#e31a1c",
    "#6a3d9a",
]


def _sample_indices(count: int, maximum: int, seed: int = 17) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(count, maximum, replace=False))


def _point_colors(colors: np.ndarray, indices: np.ndarray) -> list[str]:
    rgb = np.clip(np.asarray(colors)[indices] * 255.0, 0, 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def measured_trace(
    cloud: GeometryBundle,
    maximum: int,
    use_source_colors: bool = False,
    selectable: bool = False,
) -> go.Scatter3d:
    indices = _sample_indices(len(cloud.points), maximum)
    points = cloud.points[indices]
    colors: str | list[str] = VGGT_GREY
    if use_source_colors:
        colors = _point_colors(cloud.colors, indices)
    customdata = np.column_stack(
        [np.full(len(indices), "VGGT", dtype=object), indices.astype(object)]
    )
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name="VGGT measured",
        customdata=customdata,
        marker={
            "size": 2.0 if selectable else 1.7,
            "color": colors,
            "opacity": 0.72,
            "line": {"width": 0},
        },
        selected={"marker": {"size": 6, "color": ANCHOR_RED, "opacity": 1.0}},
        unselected={"marker": {"opacity": 0.55}},
        hovertemplate=(
            "VGGT point %{customdata[1]}<br>"
            "x=%{x:.7f}<br>y=%{y:.7f}<br>z=%{z:.7f}<extra></extra>"
        ),
    )


def mesh_trace(
    mesh: trimesh.Trimesh,
    name: str,
    color: str,
    opacity: float = 0.72,
) -> go.Mesh3d:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        flatshading=False,
        hoverinfo="name",
        lighting={"ambient": 0.55, "diffuse": 0.75, "roughness": 0.85},
    )


def layout_3d(title: str, height: int = 720) -> dict:
    return {
        "title": {"text": title, "x": 0.02, "xanchor": "left"},
        "height": height,
        "margin": {"l": 0, "r": 0, "b": 0, "t": 48},
        "scene": {
            "aspectmode": "data",
            "xaxis": {"title": "x", "showspikes": False},
            "yaxis": {"title": "y", "showspikes": False},
            "zaxis": {"title": "z", "showspikes": False},
        },
        "legend": {"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.65)"},
        "uirevision": "fuse-camera-v1",
    }


def overlay_figure(
    measured: GeometryBundle,
    prior: trimesh.Trimesh | None = None,
    missing: trimesh.Trimesh | None = None,
    manual_points: np.ndarray | None = None,
    title: str = "FUSE geometry",
    maximum_measured: int = 80_000,
    use_source_colors: bool = False,
    selectable: bool = False,
    anchor: np.ndarray | None = None,
) -> go.Figure:
    traces: list[go.BaseTraceType] = [
        measured_trace(
            measured,
            maximum_measured,
            use_source_colors=use_source_colors,
            selectable=selectable,
        )
    ]
    if prior is not None:
        traces.append(mesh_trace(prior, "Hunyuan / Kaolin hypothesis", PRIOR_CYAN, 0.48))
    if missing is not None:
        traces.append(mesh_trace(missing, "selected missing hypothesis", MISSING_YELLOW, 0.92))
    if manual_points is not None and len(manual_points):
        points = np.asarray(manual_points).reshape(-1, 3)
        traces.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers+lines",
                name="manual restoration points",
                customdata=np.column_stack(
                    [np.full(len(points), "MANUAL", dtype=object), np.arange(len(points), dtype=object)]
                ),
                marker={"size": 6, "color": MANUAL_MAGENTA, "symbol": "diamond"},
                line={"width": 3, "color": MANUAL_MAGENTA},
                hovertemplate=(
                    "manual point %{customdata[1]}<br>"
                    "x=%{x:.7f}<br>y=%{y:.7f}<br>z=%{z:.7f}<extra></extra>"
                ),
            )
        )
    if anchor is not None:
        point = np.asarray(anchor).reshape(3)
        traces.append(
            go.Scatter3d(
                x=[point[0]],
                y=[point[1]],
                z=[point[2]],
                mode="markers",
                name="current anchor",
                marker={"size": 9, "color": ANCHOR_RED, "symbol": "cross"},
                hovertemplate="anchor<br>x=%{x:.7f}<br>y=%{y:.7f}<br>z=%{z:.7f}<extra></extra>",
            )
        )
    figure = go.Figure(traces)
    figure.update_layout(**layout_3d(title))
    return figure


def candidate_figure(
    measured: GeometryBundle,
    source_mesh: trimesh.Trimesh,
    components: Iterable[CandidateComponent],
    labels: dict[int, str],
    maximum_measured: int = 50_000,
) -> go.Figure:
    traces: list[go.BaseTraceType] = [
        measured_trace(measured, maximum_measured, use_source_colors=False, selectable=False)
    ]
    for item in components:
        component = source_mesh.submesh([item.faces], append=True, repair=False)
        label = labels.get(item.component_id, "uncertain")
        if label == "missing — send to verification":
            color, opacity = MISSING_YELLOW, 0.96
        elif label == "false positive":
            color, opacity = "rgb(105,110,116)", 0.22
        else:
            color = CANDIDATE_PALETTE[item.component_id % len(CANDIDATE_PALETTE)]
            opacity = 0.84
        trace = mesh_trace(
            component,
            f"candidate {item.component_id} · {label}",
            color,
            opacity,
        )
        traces.append(trace)
        centroid = item.centroid
        traces.append(
            go.Scatter3d(
                x=[centroid[0]],
                y=[centroid[1]],
                z=[centroid[2]],
                mode="markers+text",
                name=f"ID {item.component_id}",
                text=[str(item.component_id)],
                textposition="top center",
                marker={"size": 5, "color": color, "line": {"width": 1, "color": "black"}},
                showlegend=False,
                hovertemplate=f"candidate {item.component_id}<extra></extra>",
            )
        )
    figure = go.Figure(traces)
    figure.update_layout(
        **layout_3d(
            "Candidate review — yellow=selected, grey=false positive, coloured=uncertain",
            height=760,
        )
    )
    return figure
