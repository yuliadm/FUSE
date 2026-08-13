from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import trimesh
import yaml

from fuse_manufacturing.cli import run_manufacturing
from fuse_manufacturing.mesh_ops import (
    boundary_loops,
    cap_planar,
    mesh_health,
    stitch_measured_patch,
)
from fuse_manufacturing.threemf import export_3mf


def open_cone() -> trimesh.Trimesh:
    mesh = trimesh.creation.cone(radius=1.0, height=2.0, sections=40)
    centres = np.asarray(mesh.triangles_center)
    keep = ~np.isclose(centres[:, 2], 0.0)
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return mesh


class ManufacturingTests(unittest.TestCase):
    def test_planar_cap_closes_one_boundary(self):
        source = open_cone()
        self.assertEqual(len(boundary_loops(source)), 1)
        repaired, info = cap_planar(source, max_planarity_rms_mm=1e-8)
        self.assertTrue(repaired.is_watertight)
        self.assertTrue(repaired.is_winding_consistent)
        self.assertEqual(info.method, "planar_cap")
        self.assertEqual(mesh_health(repaired)["boundary_loops"], 0)

    def test_3mf_has_required_package_parts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.3mf"
            export_3mf(trimesh.creation.box(extents=(1.0, 2.0, 3.0)), path)
            with ZipFile(path) as archive:
                self.assertIn("[Content_Types].xml", archive.namelist())
                self.assertIn("_rels/.rels", archive.namelist())
                model = archive.read("3D/3dmodel.model")
                self.assertIn(b'unit="millimeter"', model)

    def test_measured_patch_stitches_different_boundary_counts(self):
        exterior = open_cone()
        angles = np.linspace(0.0, 2.0 * np.pi, 25, endpoint=False)
        vertices = np.vstack(
            [np.zeros(3), np.column_stack([np.cos(angles), np.sin(angles), np.zeros(25)])]
        )
        faces = np.asarray(
            [[0, 1 + index, 1 + ((index + 1) % 25)] for index in range(25)],
            dtype=np.int64,
        )
        patch = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        repaired, info, shifted_patch = stitch_measured_patch(
            exterior, patch, clearance_mm=0.1, normal_sign=1
        )
        self.assertTrue(repaired.is_watertight)
        self.assertEqual(info.method, "measured_patch")
        self.assertAlmostEqual(info.clearance_mm, 0.1)
        self.assertEqual(len(boundary_loops(shifted_patch)), 1)

    def test_end_to_end_without_freecad(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            handoff = data_root / "04_verification" / "runs" / "handoff_test"
            handoff.mkdir(parents=True)
            (handoff / "handoff_manifest.json").write_text(
                json.dumps(
                    {
                        "stage": "04_human_review_and_freecad_handoff",
                        "selected_component_ids": [3],
                        "manual_restoration_points": 0,
                    }
                )
            )
            open_cone().export(handoff / "fragment_exterior_hypothesis.ply")
            config = {
                "input": {"handoff_run": "latest", "fragment_mesh": None},
                "units": {"source_to_mm": 10.0, "confirmed": True},
                "interface": {"method": "planar_cap", "max_planarity_rms_mm": 0.01},
                "checks": {
                    "require_single_component": True,
                    "require_watertight": True,
                    "require_winding_consistent": True,
                },
                "print": {"orientation": "interface_normal_to_bed", "material": "PLA"},
                "output": {"run_dir": "05_manufacturing/runs/test"},
            }
            config_path = Path(temporary) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            output = run_manufacturing(config_path, data_root, skip_freecad=True)
            self.assertTrue((output / "repair_model_mm.stl").exists())
            self.assertTrue((output / "repair_print_oriented.3mf").exists())
            self.assertTrue((output / "manufacturing_report.json").exists())
            self.assertTrue((output / "fuse_print_bundle.zip").exists())
            report = json.loads((output / "manufacturing_report.json").read_text())
            self.assertEqual(report["gates"]["watertight"], "passed")
            self.assertEqual(report["source"]["selected_component_ids"], [3])


if __name__ == "__main__":
    unittest.main()
