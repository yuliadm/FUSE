from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mapbox_earcut
import numpy as np
import trimesh


@dataclass(frozen=True)
class InterfaceInfo:
    method: str
    point_mm: np.ndarray
    normal_mm: np.ndarray
    boundary_vertices: int
    planarity_rms_mm: float
    clearance_mm: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "point_mm": self.point_mm.tolist(),
            "normal": self.normal_mm.tolist(),
            "boundary_vertices": int(self.boundary_vertices),
            "planarity_rms_mm": float(self.planarity_rms_mm),
            "clearance_mm": float(self.clearance_mm),
        }


def load_triangle_mesh(path: str | Path) -> trimesh.Trimesh:
    path = Path(path)
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No mesh geometry was loaded from {path}")
        mesh = loaded.to_mesh()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise TypeError(f"Expected a triangle mesh in {path}, got {type(loaded).__name__}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh is empty: {path}")
    return clean_mesh(mesh)


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    finite_vertices = np.isfinite(mesh.vertices).all(axis=1)
    if not finite_vertices.all():
        mesh.update_vertices(finite_vertices)
    if len(mesh.faces):
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh became empty during cleanup")
    return mesh


def boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    edges = np.asarray(mesh.edges_sorted, dtype=np.int64)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        return []

    neighbours: dict[int, list[int]] = {}
    for first, second in boundary:
        neighbours.setdefault(int(first), []).append(int(second))
        neighbours.setdefault(int(second), []).append(int(first))
    invalid = {vertex: linked for vertex, linked in neighbours.items() if len(linked) != 2}
    if invalid:
        sample = list(invalid.items())[:5]
        raise ValueError(
            "The open boundary is branched/non-manifold; each boundary vertex must have degree 2. "
            f"Examples: {sample}"
        )

    remaining = {tuple(map(int, edge)) for edge in boundary.tolist()}
    remaining |= {(second, first) for first, second in list(remaining)}
    loops: list[np.ndarray] = []
    visited_vertices: set[int] = set()
    for start in sorted(neighbours):
        if start in visited_vertices:
            continue
        loop = [start]
        previous = None
        current = start
        while True:
            candidates = [item for item in neighbours[current] if item != previous]
            if not candidates:
                raise ValueError("Open boundary traversal terminated before closing")
            next_vertex = candidates[0]
            if next_vertex == start:
                break
            if next_vertex in loop:
                raise ValueError("Open boundary self-intersects in vertex topology")
            loop.append(next_vertex)
            previous, current = current, next_vertex
        visited_vertices.update(loop)
        loops.append(np.asarray(loop, dtype=np.int64))
    loops.sort(key=len, reverse=True)
    return loops


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        raise ValueError("At least three boundary points are needed to fit a plane")
    centre = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centre, full_matrices=False)
    u_axis = vh[0]
    v_axis = vh[1]
    normal = vh[2]
    distances = (points - centre) @ normal
    rms = float(np.sqrt(np.mean(np.square(distances))))
    basis = np.column_stack([u_axis, v_axis])
    return centre, normal / np.linalg.norm(normal), basis, rms


def _fix_winding(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    trimesh.repair.fix_normals(mesh, multibody=True)
    if mesh.is_watertight and float(mesh.volume) < 0:
        mesh.invert()
    return mesh


def _area_weighted_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    normals = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    weighted = np.asarray(mesh.face_normals) * np.asarray(mesh.area_faces)[:, None]
    for corner in range(3):
        np.add.at(normals, np.asarray(mesh.faces)[:, corner], weighted)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    if not valid.all():
        normals[~valid] = np.array([0.0, 0.0, 1.0])
    return normals


def cap_planar(
    exterior: trimesh.Trimesh,
    max_planarity_rms_mm: float,
) -> tuple[trimesh.Trimesh, InterfaceInfo]:
    exterior = clean_mesh(exterior)
    loops = boundary_loops(exterior)
    if len(loops) != 1:
        raise ValueError(
            "planar_cap requires exactly one open boundary loop; "
            f"found {len(loops)}. Repair/select the candidate component first."
        )
    loop = loops[0]
    boundary_points = np.asarray(exterior.vertices)[loop]
    centre, normal, basis, rms = fit_plane(boundary_points)
    if rms > float(max_planarity_rms_mm):
        raise ValueError(
            f"Boundary planarity RMS is {rms:.4f} mm, above the configured "
            f"{max_planarity_rms_mm:.4f} mm. Use measured_patch or improve the component cut."
        )
    projected = np.ascontiguousarray((boundary_points - centre) @ basis, dtype=np.float64)
    ring_ends = np.asarray([len(projected)], dtype=np.uint32)
    local_faces = np.asarray(
        mapbox_earcut.triangulate_float64(projected, ring_ends), dtype=np.int64
    ).reshape(-1, 3)
    if len(local_faces) == 0:
        raise ValueError("Earcut could not triangulate the fragment boundary")
    cap_faces = loop[local_faces]
    result = trimesh.Trimesh(
        vertices=np.asarray(exterior.vertices).copy(),
        faces=np.vstack([np.asarray(exterior.faces), cap_faces]),
        process=False,
    )
    result = _fix_winding(clean_mesh(result))
    info = InterfaceInfo(
        method="planar_cap",
        point_mm=centre,
        normal_mm=normal,
        boundary_vertices=len(loop),
        planarity_rms_mm=rms,
        clearance_mm=0.0,
    )
    return result, info


def _loop_alignment(first: np.ndarray, second: np.ndarray) -> tuple[int, bool]:
    """Return the cyclic start and direction of second that best match first."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    sample_count = max(len(first), len(second), 32)
    first_idx = np.floor(np.arange(sample_count) * len(first) / sample_count).astype(int)
    first_sample = first[first_idx]
    best = (float("inf"), 0, False)
    for reverse in (False, True):
        candidate = second[::-1] if reverse else second
        for shift in range(len(candidate)):
            shifted = np.roll(candidate, -shift, axis=0)
            second_idx = np.floor(
                np.arange(sample_count) * len(shifted) / sample_count
            ).astype(int)
            error = float(np.mean(np.sum((first_sample - shifted[second_idx]) ** 2, axis=1)))
            if error < best[0]:
                original_shift = (len(second) - 1 - shift) % len(second) if reverse else shift
                best = (error, original_shift, reverse)
    return best[1], best[2]


def _bridge_faces(
    first_loop: np.ndarray,
    second_loop: np.ndarray,
    second_offset: int,
) -> np.ndarray:
    first_loop = np.asarray(first_loop, dtype=np.int64)
    second_loop = np.asarray(second_loop, dtype=np.int64) + int(second_offset)
    first_count = len(first_loop)
    second_count = len(second_loop)
    faces: list[list[int]] = []
    first_step = second_step = 0
    while first_step < first_count or second_step < second_count:
        first_fraction = (
            (first_step + 1) / first_count if first_step < first_count else float("inf")
        )
        second_fraction = (
            (second_step + 1) / second_count
            if second_step < second_count
            else float("inf")
        )
        first_now = int(first_loop[first_step % first_count])
        second_now = int(second_loop[second_step % second_count])
        if first_fraction <= second_fraction:
            first_next = int(first_loop[(first_step + 1) % first_count])
            faces.append([first_now, first_next, second_now])
            first_step += 1
        else:
            second_next = int(second_loop[(second_step + 1) % second_count])
            faces.append([first_now, second_next, second_now])
            second_step += 1
    return np.asarray(faces, dtype=np.int64)


def stitch_measured_patch(
    exterior: trimesh.Trimesh,
    measured_patch: trimesh.Trimesh,
    clearance_mm: float,
    normal_sign: int,
) -> tuple[trimesh.Trimesh, InterfaceInfo, trimesh.Trimesh]:
    exterior = clean_mesh(exterior)
    patch = clean_mesh(measured_patch)
    exterior_loops = boundary_loops(exterior)
    patch_loops = boundary_loops(patch)
    if len(exterior_loops) != 1 or len(patch_loops) != 1:
        raise ValueError(
            "measured_patch requires exactly one boundary loop on each mesh; "
            f"exterior={len(exterior_loops)}, patch={len(patch_loops)}"
        )
    exterior_loop = exterior_loops[0]
    patch_loop = patch_loops[0]

    sign = 1 if int(normal_sign) >= 0 else -1
    if float(clearance_mm) != 0.0:
        patch.vertices = np.asarray(patch.vertices) + (
            sign * float(clearance_mm) * _area_weighted_vertex_normals(patch)
        )

    exterior_points = np.asarray(exterior.vertices)[exterior_loop]
    patch_points = np.asarray(patch.vertices)[patch_loop]
    shift, reverse = _loop_alignment(exterior_points, patch_points)
    if reverse:
        patch_loop = patch_loop[::-1]
    patch_loop = np.roll(patch_loop, -shift)

    vertex_offset = len(exterior.vertices)
    bridge = _bridge_faces(exterior_loop, patch_loop, vertex_offset)
    result = trimesh.Trimesh(
        vertices=np.vstack([np.asarray(exterior.vertices), np.asarray(patch.vertices)]),
        faces=np.vstack(
            [
                np.asarray(exterior.faces),
                np.asarray(patch.faces) + vertex_offset,
                bridge,
            ]
        ),
        process=False,
    )
    result = _fix_winding(clean_mesh(result))
    patch_boundary = np.asarray(patch.vertices)[patch_loop]
    centre, normal, _, rms = fit_plane(patch_boundary)
    bulk_vector = np.asarray(result.centroid) - centre
    if np.dot(normal, bulk_vector) < 0:
        normal = -normal
    info = InterfaceInfo(
        method="measured_patch",
        point_mm=centre,
        normal_mm=normal,
        boundary_vertices=len(patch_loop),
        planarity_rms_mm=rms,
        clearance_mm=float(clearance_mm),
    )
    return result, info, patch


def closed_mesh_interface(mesh: trimesh.Trimesh) -> InterfaceInfo:
    if not mesh.is_watertight:
        raise ValueError("interface.method=closed_mesh requires a watertight input mesh")
    bounds = np.asarray(mesh.bounds)
    point = np.array([mesh.centroid[0], mesh.centroid[1], bounds[0, 2]], dtype=np.float64)
    return InterfaceInfo(
        method="closed_mesh",
        point_mm=point,
        normal_mm=np.array([0.0, 0.0, 1.0]),
        boundary_vertices=0,
        planarity_rms_mm=0.0,
        clearance_mm=0.0,
    )


def align_vector_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    transform = np.eye(4)
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0:
            return transform
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis -= source * np.dot(axis, source)
        axis /= np.linalg.norm(axis)
        rotation = -np.eye(3) + 2.0 * np.outer(axis, axis)
        transform[:3, :3] = rotation
        return transform
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    rotation = np.eye(3) + skew + (skew @ skew) * ((1.0 - dot) / np.dot(cross, cross))
    transform[:3, :3] = rotation
    return transform


def orient_for_print(
    mesh: trimesh.Trimesh,
    interface: InterfaceInfo,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    result = mesh.copy()
    bulk_vector = np.asarray(mesh.centroid) - interface.point_mm
    normal = np.asarray(interface.normal_mm, dtype=np.float64)
    if np.dot(normal, bulk_vector) < 0:
        normal = -normal
    transform = align_vector_matrix(normal, np.array([0.0, 0.0, 1.0]))
    result.apply_transform(transform)
    translation = np.eye(4)
    translation[2, 3] = -float(result.bounds[0, 2])
    result.apply_transform(translation)
    return result, translation @ transform


def mesh_health(mesh: trimesh.Trimesh) -> dict[str, Any]:
    loops = boundary_loops(mesh)
    parts = mesh.split(only_watertight=False)
    edge_lengths = np.asarray(mesh.edges_unique_length, dtype=np.float64)
    result: dict[str, Any] = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(parts)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "boundary_loops": int(len(loops)),
        "bounds_mm": np.asarray(mesh.bounds).tolist(),
        "extents_mm": np.asarray(mesh.extents).tolist(),
        "surface_area_mm2": float(mesh.area),
        "edge_length_mm": {
            "min": float(np.min(edge_lengths)),
            "median": float(np.median(edge_lengths)),
            "max": float(np.max(edge_lengths)),
        },
    }
    if mesh.is_watertight:
        result["volume_mm3"] = float(abs(mesh.volume))
        result["mass_pla_g"] = float(abs(mesh.volume) * 1.24e-3)
    else:
        result["volume_mm3"] = None
        result["mass_pla_g"] = None
    return result


def sampled_vertex_proximity(
    fragment: trimesh.Trimesh,
    broken: trimesh.Trimesh,
    sample_limit: int = 5000,
) -> dict[str, float]:
    rng = np.random.default_rng(20260809)
    fragment_vertices = np.asarray(fragment.vertices)
    broken_vertices = np.asarray(broken.vertices)
    if len(fragment_vertices) > sample_limit:
        fragment_vertices = fragment_vertices[
            rng.choice(len(fragment_vertices), sample_limit, replace=False)
        ]
    if len(broken_vertices) > sample_limit:
        broken_vertices = broken_vertices[rng.choice(len(broken_vertices), sample_limit, replace=False)]
    minima = np.full(len(fragment_vertices), np.inf)
    for start in range(0, len(broken_vertices), 500):
        block = broken_vertices[start : start + 500]
        distances2 = np.sum(
            (fragment_vertices[:, None, :] - block[None, :, :]) ** 2, axis=2
        )
        minima = np.minimum(minima, np.sqrt(np.min(distances2, axis=1)))
    return {
        "method": "sampled_vertex_to_vertex; diagnostic only",
        "minimum_mm": float(np.min(minima)),
        "p01_mm": float(np.quantile(minima, 0.01)),
        "median_mm": float(np.median(minima)),
    }
