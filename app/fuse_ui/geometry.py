from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
import csv
import hashlib
import json
import zipfile

import numpy as np
import open3d as o3d
import trimesh


STATE_COLORS = np.asarray(
    [
        [40, 190, 90],   # supported
        [245, 165, 35],  # uncertain
        [225, 60, 70],   # unsupported
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class GeometryBundle:
    points: np.ndarray
    colors: np.ndarray
    path: Path

    @property
    def diagonal(self) -> float:
        if len(self.points) == 0:
            return 0.0
        return float(np.linalg.norm(np.ptp(self.points, axis=0)))


@dataclass(frozen=True)
class CandidateComponent:
    component_id: int
    faces: np.ndarray
    vertices: np.ndarray
    area: float
    centroid: np.ndarray


def _existing_path(value: str | Path | None, fuse_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    if path.is_absolute() and str(path).startswith("/workspace/"):
        rebased = fuse_root / path.relative_to("/workspace")
        if rebased.exists():
            return rebased
    if not path.is_absolute():
        candidate = fuse_root / path
        if candidate.exists():
            return candidate
    return None


def discover_alignment_runs(data_root: Path) -> list[Path]:
    runs_root = Path(data_root) / "kaolin_outputs" / "alignment" / "runs"
    if not runs_root.exists():
        return []
    runs = [path for path in runs_root.iterdir() if path.is_dir()]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def discover_inputs(fuse_root: Path, run_dir: Path) -> dict[str, Path | None]:
    fuse_root = Path(fuse_root)
    data_root = fuse_root / "data"
    report_path = run_dir / "alignment_report.json"
    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except (json.JSONDecodeError, OSError):
            report = {}

    inputs = report.get("inputs", {})
    outputs = report.get("outputs", {})

    measured = _existing_path(inputs.get("vggt_cloud"), fuse_root)
    if measured is None:
        preferred = [
            data_root / "vggt_outputs" / "global" / "broken_clean_normals.ply",
            data_root / "vggt_outputs" / "global" / "broken_clean.ply",
            data_root / "vggt_outputs" / "global" / "cleaned_cloud.ply",
            data_root / "vggt_outputs" / "global" / "raw_vggt_cloud.ply",
        ]
        measured = next((path for path in preferred if path.exists()), None)

    prior = _existing_path(outputs.get("adapted_prior"), fuse_root)
    if prior is None:
        prior = next(
            (
                path
                for path in (
                    run_dir / "prior_adapted.glb",
                    run_dir / "prior_kaolin_aligned.glb",
                    run_dir / "prior_coarse_sim3.glb",
                )
                if path.exists()
            ),
            None,
        )

    missing = _existing_path(outputs.get("missing_piece"), fuse_root)
    if missing is None:
        missing = next(
            (
                path
                for path in (
                    run_dir / "missing_piece_hypothesis.glb",
                    run_dir / "missing_piece_hypothesis.ply",
                )
                if path.exists()
            ),
            None,
        )

    classification = run_dir / "prior_support_classification.ply"
    if not classification.exists():
        classification = None

    return {
        "report": report_path if report_path.exists() else None,
        "measured": measured,
        "prior": prior,
        "missing": missing,
        "classification": classification,
    }


def load_point_cloud(path: str | Path) -> GeometryBundle:
    path = Path(path)
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"No points were loaded from {path}")
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if cloud.has_colors():
        colors = np.asarray(cloud.colors, dtype=np.float64)[finite]
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = np.full((len(points), 3), 0.52, dtype=np.float64)
    return GeometryBundle(points=points, colors=colors, path=path)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    path = Path(path)
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry was loaded from {path}")
        try:
            mesh = loaded.dump(concatenate=True)
        except Exception:
            mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, got {type(mesh).__name__}")
    mesh = mesh.copy()
    finite = np.isfinite(mesh.vertices).all(axis=1)
    if not finite.all():
        mesh.update_vertices(finite)
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"The mesh is empty after loading {path}")
    return mesh


def _vertex_states_from_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None or len(colors) != len(mesh.vertices):
        raise ValueError(
            "The support-classification PLY has no per-vertex colors. "
            "Rerun the notebook classification cell."
        )
    rgb = np.asarray(colors, dtype=np.float64)[:, :3]
    distance2 = np.sum((rgb[:, None, :] - STATE_COLORS[None, :, :]) ** 2, axis=2)
    states = np.argmin(distance2, axis=1).astype(np.uint8)
    if float(np.median(np.min(distance2, axis=1))) > 36.0:
        raise ValueError("The PLY colors do not match the FUSE support-state palette.")
    return states


def build_candidate_components(
    classification_mesh: trimesh.Trimesh,
    min_faces: int = 40,
) -> list[CandidateComponent]:
    """Reconstruct the notebook's unsupported connected components and IDs."""
    states = _vertex_states_from_colors(classification_mesh)
    faces = np.asarray(classification_mesh.faces, dtype=np.int64)
    face_states = states[faces]
    candidate_mask = (
        (np.sum(face_states == 2, axis=1) >= 2)
        & (np.sum(face_states == 0, axis=1) == 0)
    )
    candidate_faces = np.flatnonzero(candidate_mask)
    if len(candidate_faces) == 0:
        return []

    adjacency = np.asarray(classification_mesh.face_adjacency, dtype=np.int64)
    if len(adjacency):
        keep = candidate_mask[adjacency[:, 0]] & candidate_mask[adjacency[:, 1]]
        adjacency = adjacency[keep]
    components = trimesh.graph.connected_components(
        adjacency,
        nodes=candidate_faces,
        min_len=max(1, int(min_faces)),
    )

    records: list[CandidateComponent] = []
    area_faces = np.asarray(classification_mesh.area_faces)
    centers = np.asarray(classification_mesh.triangles_center)
    for component_faces in components:
        component_faces = np.asarray(component_faces, dtype=np.int64)
        vertices = np.unique(faces[component_faces].ravel())
        weights = area_faces[component_faces]
        area = float(weights.sum())
        centroid = np.average(
            centers[component_faces],
            axis=0,
            weights=np.maximum(weights, 1e-12),
        )
        records.append(
            CandidateComponent(
                component_id=-1,
                faces=component_faces,
                vertices=vertices,
                area=area,
                centroid=np.asarray(centroid, dtype=np.float64),
            )
        )

    records.sort(key=lambda item: item.area, reverse=True)
    return [
        CandidateComponent(
            component_id=index,
            faces=item.faces,
            vertices=item.vertices,
            area=item.area,
            centroid=item.centroid,
        )
        for index, item in enumerate(records)
    ]


def merge_candidate_components(
    source_mesh: trimesh.Trimesh,
    components: Iterable[CandidateComponent],
    selected_ids: Iterable[int],
) -> trimesh.Trimesh | None:
    wanted = {int(value) for value in selected_ids}
    face_groups = [item.faces for item in components if item.component_id in wanted]
    if not face_groups:
        return None
    selected_faces = np.concatenate(face_groups)
    result = source_mesh.submesh([selected_faces], append=True, repair=False)
    result.remove_unreferenced_vertices()
    return result


def _as_bytes(value: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def export_mesh_bytes(mesh: trimesh.Trimesh, file_type: str) -> bytes:
    return _as_bytes(mesh.export(file_type=file_type))


def point_cloud_ply_bytes(
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> bytes:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if colors is None:
        colors_u8 = np.full((len(points), 3), 255, dtype=np.uint8)
    else:
        colors_array = np.asarray(colors).reshape(-1, 3)
        if len(colors_array) != len(points):
            raise ValueError("Point and color counts must match")
        if np.issubdtype(colors_array.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(colors_array, initial=0.0)) <= 1.0 else 1.0
            colors_u8 = np.clip(colors_array * scale, 0, 255).astype(np.uint8)
        else:
            colors_u8 = np.clip(colors_array, 0, 255).astype(np.uint8)

    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(
        f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g} {int(c[0])} {int(c[1])} {int(c[2])}"
        for p, c in zip(points, colors_u8)
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def manual_points_csv_bytes(points: np.ndarray) -> bytes:
    output = BytesIO()
    text = output.write
    text(b"point_id,x,y,z,role\n")
    for index, point in enumerate(np.asarray(points).reshape(-1, 3)):
        text(
            f"{index},{point[0]:.12g},{point[1]:.12g},{point[2]:.12g},manual_restoration\n".encode(
                "utf-8"
            )
        )
    return output.getvalue()


def save_fragment_edit(
    data_root: Path,
    run_dir: Path,
    labels: dict[int, str],
    selected_mesh: trimesh.Trimesh | None,
    manual_points: np.ndarray,
) -> Path:
    manual_points = np.asarray(manual_points, dtype=np.float64).reshape(-1, 3)
    base_points = (
        np.empty((0, 3), dtype=np.float64)
        if selected_mesh is None
        else np.asarray(selected_mesh.vertices, dtype=np.float64).reshape(-1, 3)
    )
    if not len(base_points) and not len(manual_points):
        raise ValueError("A fragment edit needs selected or manually authored points")

    selected_ids = sorted(
        int(component_id)
        for component_id, label in labels.items()
        if label == "missing — send to verification"
    )
    if len(base_points) and len(manual_points):
        mode = "selected_plus_manual"
    elif len(manual_points):
        mode = "manual_only"
    else:
        mode = "selected_only"

    timestamp = datetime.now(timezone.utc).strftime("fragment_edit_%Y%m%dT%H%M%S%fZ")
    output_dir = Path(data_root) / "kaolin_outputs" / "fragment_edits" / "runs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    artifacts: dict[str, Any] = {}
    if len(manual_points):
        manual_bytes = point_cloud_ply_bytes(
            manual_points,
            np.tile(np.asarray([[255, 77, 157]], dtype=np.uint8), (len(manual_points), 1)),
        )
        manual_name = "manual_fragment_points.ply"
        (output_dir / manual_name).write_bytes(manual_bytes)
        artifacts["manual_points"] = {
            "path": manual_name,
            "sha256": hashlib.sha256(manual_bytes).hexdigest(),
            "point_count": int(len(manual_points)),
        }

    edited_points = np.vstack([base_points, manual_points])
    edited_colors = np.vstack(
        [
            np.tile(np.asarray([[255, 215, 40]], dtype=np.uint8), (len(base_points), 1)),
            np.tile(np.asarray([[255, 77, 157]], dtype=np.uint8), (len(manual_points), 1)),
        ]
    )
    edited_bytes = point_cloud_ply_bytes(edited_points, edited_colors)
    edited_name = "edited_fragment_points.ply"
    (output_dir / edited_name).write_bytes(edited_bytes)
    artifacts["working_fragment_points"] = {
        "path": edited_name,
        "sha256": hashlib.sha256(edited_bytes).hexdigest(),
        "point_count": int(len(edited_points)),
    }

    manifest = {
        "schema": "fuse.fragment-edit/v1",
        "stage": "03_fragment_edit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_alignment_run": str(run_dir),
        "authority_rule": "VGGT measured geometry remains unchanged.",
        "units": "VGGT coordinate units; not assumed to be millimetres",
        "mode": mode,
        "selected_component_ids": selected_ids,
        "base_fragment": {
            "source": (
                "human_selected_kaolin_components"
                if selected_ids
                else "existing_missing_piece_export"
                if len(base_points)
                else None
            ),
            "point_count": int(len(base_points)),
            "provenance": "inferred",
        },
        "manual_fragment": {
            "point_count": int(len(manual_points)),
            "provenance": "human_authored",
        },
        "working_fragment": {
            "point_count": int(len(edited_points)),
            "status": "valid_stage_3_handoff",
            "surface_status": "not_constructed" if selected_mesh is None else "candidate_mesh_available",
        },
        "artifacts": artifacts,
    }
    (output_dir / "fragment_edit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    run_dir: Path,
    labels: dict[int, str],
    selected_mesh: trimesh.Trimesh | None,
    manual_points: np.ndarray,
) -> dict[str, Any]:
    selected_ids = sorted(
        int(component_id)
        for component_id, label in labels.items()
        if label == "missing — send to verification"
    )
    return {
        "stage": "04_human_review_and_freecad_handoff",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_alignment_run": str(run_dir),
        "authority_rule": "VGGT measured geometry remains unchanged.",
        "units": "VGGT coordinate units; not assumed to be millimetres",
        "candidate_labels": {str(key): value for key, value in sorted(labels.items())},
        "selected_component_ids": selected_ids,
        "manual_restoration_points": int(len(manual_points)),
        "fragment_health": None
        if selected_mesh is None
        else {
            "vertices": int(len(selected_mesh.vertices)),
            "faces": int(len(selected_mesh.faces)),
            "watertight": bool(selected_mesh.is_watertight),
            "winding_consistent": bool(selected_mesh.is_winding_consistent),
            "euler_number": int(selected_mesh.euler_number),
            "area": float(selected_mesh.area),
        },
        "manufacturing_gate": {
            "status": "verification_required",
            "reason": (
                "Selected components describe an exterior hypothesis. Build a measured mating "
                "surface, apply clearance, close the solid, set physical scale, and verify fit "
                "before printing."
            ),
            "stl_included": bool(selected_mesh is not None and selected_mesh.is_watertight),
        },
    }


def build_handoff_zip(
    run_dir: Path,
    labels: dict[int, str],
    selected_mesh: trimesh.Trimesh | None,
    manual_points: np.ndarray,
) -> bytes:
    manual_points = np.asarray(manual_points, dtype=np.float64).reshape(-1, 3)
    manifest = build_manifest(run_dir, labels, selected_mesh, manual_points)
    output = BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("handoff_manifest.json", json.dumps(manifest, indent=2))
        archive.writestr(
            "README_FREECAD.txt",
            "FUSE FreeCAD / verification hand-off\n\n"
            "The mesh is an exterior missing-fragment hypothesis in VGGT coordinate units.\n"
            "It is not printable until the fracture mating surface and clearance are built, the\n"
            "mesh is closed, physical scale is confirmed, and fit is verified. Import the OBJ\n"
            "or GLB for inspection. Use STL only when the manifest says watertight=true.\n",
        )
        if selected_mesh is not None:
            archive.writestr(
                "fragment_exterior_hypothesis.glb",
                export_mesh_bytes(selected_mesh, "glb"),
            )
            archive.writestr(
                "fragment_exterior_hypothesis.ply",
                export_mesh_bytes(selected_mesh, "ply"),
            )
            archive.writestr(
                "fragment_exterior_hypothesis.obj",
                export_mesh_bytes(selected_mesh, "obj"),
            )
            if selected_mesh.is_watertight:
                archive.writestr(
                    "fragment_exterior_hypothesis.stl",
                    export_mesh_bytes(selected_mesh, "stl"),
                )
        if len(manual_points):
            archive.writestr(
                "manual_restoration_points.csv",
                manual_points_csv_bytes(manual_points),
            )
            archive.writestr(
                "manual_restoration_points.ply",
                point_cloud_ply_bytes(
                    manual_points,
                    np.tile(np.asarray([[255, 77, 157]], dtype=np.uint8), (len(manual_points), 1)),
                ),
            )
    return output.getvalue()


def save_handoff(
    data_root: Path,
    run_dir: Path,
    labels: dict[int, str],
    selected_mesh: trimesh.Trimesh | None,
    manual_points: np.ndarray,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("handoff_%Y%m%dT%H%M%SZ")
    output_dir = Path(data_root) / "kaolin_outputs"/ "verification" / "runs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    manual_points = np.asarray(manual_points, dtype=np.float64).reshape(-1, 3)
    manifest = build_manifest(run_dir, labels, selected_mesh, manual_points)
    (output_dir / "handoff_manifest.json").write_text(json.dumps(manifest, indent=2))
    if selected_mesh is not None:
        selected_mesh.export(output_dir / "fragment_exterior_hypothesis.glb")
        selected_mesh.export(output_dir / "fragment_exterior_hypothesis.ply")
        selected_mesh.export(output_dir / "fragment_exterior_hypothesis.obj")
        if selected_mesh.is_watertight:
            selected_mesh.export(output_dir / "fragment_exterior_hypothesis.stl")
    if len(manual_points):
        (output_dir / "manual_restoration_points.csv").write_bytes(
            manual_points_csv_bytes(manual_points)
        )
        (output_dir / "manual_restoration_points.ply").write_bytes(
            point_cloud_ply_bytes(
                manual_points,
                np.tile(np.asarray([[255, 77, 157]], dtype=np.uint8), (len(manual_points), 1)),
            )
        )
    return output_dir
