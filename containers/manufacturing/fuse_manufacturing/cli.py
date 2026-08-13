from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import trimesh
import yaml

from .mesh_ops import (
    cap_planar,
    clean_mesh,
    closed_mesh_interface,
    load_triangle_mesh,
    mesh_health,
    orient_for_print,
    sampled_vertex_proximity,
    stitch_measured_patch,
)
from .threemf import export_3mf


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = PACKAGE_ROOT / "manufacturing.example.yaml"
FREECAD_SCRIPT = PACKAGE_ROOT / "freecad" / "build_document.py"


def utc_stamp(prefix: str) -> str:
    return prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return data


def resolve_handoff(data_root: Path, value: str | None) -> Path:
    runs_root = data_root / "04_verification" / "runs"
    if not value or value == "latest":
        candidates = sorted(
            (item for item in runs_root.glob("handoff_*") if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No Stage 4 hand-off run exists under {runs_root}")
        return candidates[0]
    path = Path(value)
    if not path.is_absolute():
        path = data_root / value
    if not path.is_dir():
        raise FileNotFoundError(f"Hand-off directory does not exist: {path}")
    return path


def resolve_file(value: str | None, data_root: Path, handoff: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates = [handoff / path, data_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Configured file was not found: {value}")


def discover_fragment(handoff: Path, configured: str | None, data_root: Path) -> Path:
    explicit = resolve_file(configured, data_root, handoff)
    if explicit is not None:
        return explicit
    for name in (
        "fragment_exterior_hypothesis.glb",
        "fragment_exterior_hypothesis.obj",
        "fragment_exterior_hypothesis.ply",
        "fragment_exterior_hypothesis.stl",
    ):
        candidate = handoff / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No fragment exterior mesh was found in {handoff}")


def physical_scale(units: dict) -> float:
    direct = units.get("source_to_mm")
    source_distance = units.get("known_distance_source_units")
    target_distance = units.get("known_distance_mm")
    if direct is not None:
        scale = float(direct)
    elif source_distance is not None and target_distance is not None:
        source_distance = float(source_distance)
        target_distance = float(target_distance)
        if source_distance <= 0 or target_distance <= 0:
            raise ValueError("Known distances must be positive")
        scale = target_distance / source_distance
    else:
        raise ValueError(
            "Physical scale is unresolved. Set units.source_to_mm, or both known-distance fields."
        )
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("units.source_to_mm must be a finite positive number")
    if not bool(units.get("confirmed", False)):
        raise ValueError(
            "Physical scale has not been confirmed. Set units.confirmed=true only after measuring "
            "a known distance on the real figurine."
        )
    return scale


def apply_scale(mesh: trimesh.Trimesh, scale_to_mm: float) -> trimesh.Trimesh:
    result = mesh.copy()
    result.vertices = np.asarray(result.vertices, dtype=np.float64) * float(scale_to_mm)
    return clean_mesh(result)


def enforce_checks(health: dict, checks: dict) -> list[str]:
    failures: list[str] = []
    if bool(checks.get("require_single_component", True)) and health["components"] != 1:
        failures.append(f"expected one connected component, found {health['components']}")
    if bool(checks.get("require_watertight", True)) and not health["watertight"]:
        failures.append("mesh is not watertight")
    if bool(checks.get("require_winding_consistent", True)) and not health["winding_consistent"]:
        failures.append("triangle winding is inconsistent")
    if health.get("volume_mm3") is not None and health["volume_mm3"] <= 0:
        failures.append("mesh has non-positive volume")
    return failures


def write_readme(path: Path, report: dict) -> None:
    interface_method = report["interface"]["method"]
    prototype_warning = ""
    if interface_method == "planar_cap":
        prototype_warning = (
            "\nIMPORTANT: the mating face is a planar approximation. It is suitable for a "
            "prototype fit test, not a claim of fracture-exact restoration.\n"
        )
    path.write_text(
        "FUSE manufacturing bundle\n"
        "=========================\n\n"
        "repair.FCStd                FreeCAD document with source and print-oriented meshes\n"
        "repair_model_mm.stl         Closed fragment in the aligned object frame\n"
        "repair_print_oriented.stl   Closed fragment placed on Z=0 for slicing\n"
        "repair_print_oriented.3mf   Same print mesh with millimetre units embedded\n"
        "repair.step                 Editable CAD solid when BREP conversion succeeded\n"
        "manufacturing_report.json   Geometry, scale, interface, and export checks\n"
        "print_settings.json         Recommended starting settings; review in your slicer\n"
        f"{prototype_warning}\n"
        "No G-code is included. G-code is printer-, nozzle-, firmware-, and material-profile "
        "specific; generate it in the slicer configured for the actual printer.\n",
        encoding="utf-8",
    )


def package_bundle(run_dir: Path) -> Path:
    bundle = run_dir / "fuse_print_bundle.zip"
    names = [
        "README_PRINT.txt",
        "repair.FCStd",
        "repair_model_mm.stl",
        "repair_print_oriented.stl",
        "repair_print_oriented.3mf",
        "repair.step",
        "repair.brep",
        "manufacturing_report.json",
        "manufacturing_config_resolved.yaml",
        "print_settings.json",
        "handoff_manifest.json",
    ]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = run_dir / name
            if path.exists():
                archive.write(path, arcname=name)
    return bundle


def run_freecad(run_dir: Path, checks: dict) -> dict:
    job = {
        "input_stl": str(run_dir / "repair_model_mm.stl"),
        "print_stl": str(run_dir / "repair_print_oriented.stl"),
        "output_fcstd": str(run_dir / "repair.FCStd"),
        "output_step": str(run_dir / "repair.step"),
        "output_brep": str(run_dir / "repair.brep"),
        "result_json": str(run_dir / "freecad_result.json"),
        "metadata_json": str(run_dir / "manufacturing_report.json"),
        "mesh_tolerance_mm": float(checks.get("freecad_mesh_tolerance_mm", 0.03)),
        "max_brep_faces": int(checks.get("max_brep_faces", 200000)),
    }
    job_path = run_dir / "freecad_job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    command = os.environ.get("FUSE_FREECAD_CMD", "/opt/freecad/AppRun")
    process = subprocess.run(
        [command, "freecadcmd", str(FREECAD_SCRIPT)],
        env={**os.environ, "FUSE_FREECAD_JOB": str(job_path)},
        text=True,
        capture_output=True,
        check=False,
    )
    (run_dir / "freecad_stdout.log").write_text(process.stdout, encoding="utf-8")
    (run_dir / "freecad_stderr.log").write_text(process.stderr, encoding="utf-8")
    result_path = run_dir / "freecad_result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    result["process_returncode"] = process.returncode
    if process.returncode != 0 or not (run_dir / "repair.FCStd").exists():
        raise RuntimeError(
            "FreeCAD document generation failed. Inspect freecad_stdout.log and "
            f"freecad_stderr.log in {run_dir}"
        )
    return result


def run_manufacturing(config_path: Path, data_root: Path, skip_freecad: bool = False) -> Path:
    config = load_yaml(config_path)
    input_cfg = config.get("input", {})
    interface_cfg = config.get("interface", {})
    checks = config.get("checks", {})
    output_cfg = config.get("output", {})

    handoff = resolve_handoff(data_root, input_cfg.get("handoff_run"))
    manifest_path = handoff / "handoff_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage 4 decision manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest.get("selected_component_ids", [])
    if not selected and int(manifest.get("manual_restoration_points", 0)) == 0:
        raise ValueError("The Stage 4 hand-off contains no human-approved fragment hypothesis")

    fragment_path = discover_fragment(handoff, input_cfg.get("fragment_mesh"), data_root)
    scale_to_mm = physical_scale(config.get("units", {}))
    source_mesh = load_triangle_mesh(fragment_path)
    scaled_exterior = apply_scale(source_mesh, scale_to_mm)

    method = str(interface_cfg.get("method", "planar_cap"))
    measured_patch = None
    if method == "planar_cap":
        final_mesh, interface = cap_planar(
            scaled_exterior,
            max_planarity_rms_mm=float(interface_cfg.get("max_planarity_rms_mm", 0.35)),
        )
    elif method == "measured_patch":
        patch_path = resolve_file(interface_cfg.get("measured_patch_mesh"), data_root, handoff)
        if patch_path is None:
            raise ValueError("interface.measured_patch_mesh is required for measured_patch")
        patch_source = apply_scale(load_triangle_mesh(patch_path), scale_to_mm)
        final_mesh, interface, measured_patch = stitch_measured_patch(
            scaled_exterior,
            patch_source,
            clearance_mm=float(interface_cfg.get("clearance_mm", 0.20)),
            normal_sign=int(interface_cfg.get("measured_patch_normal_sign", 1)),
        )
    elif method == "closed_mesh":
        final_mesh = scaled_exterior
        interface = closed_mesh_interface(final_mesh)
    else:
        raise ValueError(f"Unsupported interface.method={method!r}")

    final_mesh = clean_mesh(final_mesh)
    health = mesh_health(final_mesh)
    failures = enforce_checks(health, checks)
    if failures:
        raise ValueError("Manufacturing geometry gate failed: " + "; ".join(failures))

    if str(config.get("print", {}).get("orientation")) == "interface_normal_to_bed":
        print_mesh, print_transform = orient_for_print(final_mesh, interface)
    else:
        print_mesh = final_mesh.copy()
        print_transform = np.eye(4)

    configured_run = output_cfg.get("run_dir")
    if configured_run:
        run_dir = Path(configured_run)
        if not run_dir.is_absolute():
            run_dir = data_root / run_dir
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Configured output directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = data_root / "05_manufacturing" / "runs" / utc_stamp("mfg_")
        run_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(manifest_path, run_dir / "handoff_manifest.json")
    shutil.copy2(config_path, run_dir / "manufacturing_config_resolved.yaml")
    final_mesh.export(run_dir / "repair_model_mm.stl")
    final_mesh.export(run_dir / "repair_model_mm.ply")
    print_mesh.export(run_dir / "repair_print_oriented.stl")
    export_3mf(print_mesh, run_dir / "repair_print_oriented.3mf")
    if measured_patch is not None:
        measured_patch.export(run_dir / "mating_patch_clearanced_mm.stl")

    broken_path = resolve_file(input_cfg.get("broken_object_mesh"), data_root, handoff)
    proximity = None
    if broken_path is not None:
        broken_mesh = apply_scale(load_triangle_mesh(broken_path), scale_to_mm)
        proximity = sampled_vertex_proximity(final_mesh, broken_mesh)

    print_settings = dict(config.get("print", {}))
    print_settings["status"] = "starting recommendations; must be reviewed in the printer's slicer"
    (run_dir / "print_settings.json").write_text(
        json.dumps(print_settings, indent=2), encoding="utf-8"
    )

    report = {
        "stage": "05_manufacturing_preparation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "geometry_ready_for_slicing",
        "fit_status": (
            "prototype_planar_interface_requires_physical_test"
            if method == "planar_cap"
            else "measured_interface_requires_physical_test"
            if method == "measured_patch"
            else "closed_input_requires_physical_test"
        ),
        "source": {
            "handoff_run": str(handoff),
            "fragment_mesh": str(fragment_path),
            "selected_component_ids": selected,
        },
        "units": {"coordinate_unit": "millimetre", "source_to_mm": scale_to_mm, "confirmed": True},
        "interface": interface.as_dict(),
        "geometry_checks": health,
        "proximity_to_broken_object": proximity,
        "print_orientation_transform": print_transform.tolist(),
        "gates": {
            "physical_scale": "passed",
            "single_component": "passed" if health["components"] == 1 else "not_required",
            "watertight": "passed" if health["watertight"] else "not_required",
            "winding_consistent": "passed" if health["winding_consistent"] else "not_required",
            "physical_fit_test": "required",
            "slicer_profile": "required_before_gcode",
        },
        "limitations": [
            "The missing geometry is a reconstruction hypothesis, not recovered ground truth.",
            "A physical dry-fit is required before adhesive use.",
            "G-code is intentionally omitted because it is printer-profile-specific.",
        ],
    }
    if method == "planar_cap":
        report["limitations"].append(
            "The planar cap approximates the fracture interface and does not reproduce its microgeometry."
        )
    report_path = run_dir / "manufacturing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if skip_freecad:
        freecad_result = {"skipped": True}
    else:
        freecad_result = run_freecad(run_dir, checks)
    report["freecad"] = freecad_result

    outputs = {}
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name not in {"manufacturing_report.json", "fuse_print_bundle.zip"}:
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    report["outputs"] = outputs
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_readme(run_dir / "README_PRINT.txt", report)
    package_bundle(run_dir)
    return run_dir


def inspect_handoff(handoff_value: str, data_root: Path) -> dict:
    handoff = resolve_handoff(data_root, handoff_value)
    manifest_path = handoff / "handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    fragment = discover_fragment(handoff, None, data_root)
    mesh = load_triangle_mesh(fragment)
    return {
        "handoff": str(handoff),
        "fragment": str(fragment),
        "manifest": manifest,
        "source_unit_health": mesh_health(mesh),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fuse-manufacturing",
        description="Create checked FreeCAD/STL/3MF manufacturing artifacts from a FUSE Stage 4 hand-off.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("FUSE_DATA_ROOT", "/workspace/data")),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-config", help="Write an editable manufacturing YAML")
    init_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/data/05_manufacturing/manufacturing.yaml"),
    )

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a Stage 4 hand-off without writing")
    inspect_parser.add_argument("--handoff", default="latest")

    run_parser = subparsers.add_parser("run", help="Run manufacturing preparation")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/data/05_manufacturing/manufacturing.yaml"),
    )
    run_parser.add_argument("--skip-freecad", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init-config":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.exists():
                raise FileExistsError(f"Refusing to overwrite existing config: {args.output}")
            shutil.copy2(EXAMPLE_CONFIG, args.output)
            print(args.output)
        elif args.command == "inspect":
            print(json.dumps(inspect_handoff(args.handoff, args.data_root), indent=2))
        elif args.command == "run":
            output = run_manufacturing(args.config, args.data_root, args.skip_freecad)
            print(output)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
