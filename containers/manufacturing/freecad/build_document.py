"""Executed by FreeCADCmd. Deliberately calls run() at import time.

FreeCADCmd imports scripts passed on its command line in some releases, so a normal
``if __name__ == '__main__'`` guard can silently skip the job.
"""

import json
import os
from pathlib import Path
import traceback

import FreeCAD as App
import Mesh
import Part


def add_string_property(obj, name, value):
    obj.addProperty("App::PropertyString", name, "FUSE")
    setattr(obj, name, str(value))


def mesh_feature(document, name, label, path):
    obj = document.addObject("Mesh::Feature", name)
    obj.Label = label
    obj.Mesh = Mesh.Mesh(str(path))
    return obj


def convert_to_solid(mesh, tolerance):
    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh.Topology, float(tolerance))
    shells = list(shape.Shells)
    if len(shells) == 1:
        solid = Part.makeSolid(shells[0])
    else:
        shell = Part.makeShell(shape.Faces)
        solid = Part.makeSolid(shell)
    if solid.isNull() or not solid.isValid():
        raise ValueError("OpenCASCADE produced an invalid solid from the triangle mesh")
    return solid.removeSplitter()


def run():
    job_path = Path(os.environ["FUSE_FREECAD_JOB"])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_path = Path(job["result_json"])
    result = {
        "freecad_version": ".".join(App.Version()[:3]),
        "fcstd_created": False,
        "brep_conversion_attempted": False,
        "brep_conversion_succeeded": False,
        "step_created": False,
        "brep_created": False,
    }
    try:
        document = App.newDocument("FUSE_Repair")
        original = mesh_feature(
            document,
            "FragmentOriginalFrame",
            "FUSE repair — aligned object frame (mm)",
            job["input_stl"],
        )
        oriented = mesh_feature(
            document,
            "FragmentPrintOriented",
            "FUSE repair — print orientation (mm)",
            job["print_stl"],
        )

        metadata = document.addObject("App::FeaturePython", "FUSEManufacturingMetadata")
        metadata.Label = "FUSE manufacturing metadata"
        add_string_property(metadata, "ReportPath", job["metadata_json"])
        add_string_property(metadata, "LengthUnit", "millimetre")
        add_string_property(metadata, "SourceModel", job["input_stl"])
        add_string_property(metadata, "PrintModel", job["print_stl"])
        metadata.addProperty("App::PropertyInteger", "TriangleFaces", "FUSE")
        metadata.TriangleFaces = int(original.Mesh.CountFacets)
        metadata.addProperty("App::PropertyBool", "PrintMeshWatertight", "FUSE")
        metadata.PrintMeshWatertight = True

        face_count = int(original.Mesh.CountFacets)
        if face_count <= int(job["max_brep_faces"]):
            result["brep_conversion_attempted"] = True
            try:
                solid_shape = convert_to_solid(original.Mesh, job["mesh_tolerance_mm"])
                solid = document.addObject("Part::Feature", "FragmentSolid")
                solid.Label = "FUSE repair — editable BREP solid"
                solid.Shape = solid_shape
                Part.export([solid], job["output_step"])
                solid.Shape.exportBrep(job["output_brep"])
                result["brep_conversion_succeeded"] = True
                result["step_created"] = Path(job["output_step"]).exists()
                result["brep_created"] = Path(job["output_brep"]).exists()
            except Exception as exc:
                result["brep_error"] = str(exc)
                result["brep_traceback"] = traceback.format_exc()
        else:
            result["brep_skip_reason"] = (
                f"{face_count} faces exceeds max_brep_faces={job['max_brep_faces']}"
            )

        document.recompute()
        document.saveAs(job["output_fcstd"])
        result["fcstd_created"] = Path(job["output_fcstd"]).exists()
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        App.closeDocument(document.Name)
    except Exception as exc:
        result["fatal_error"] = str(exc)
        result["fatal_traceback"] = traceback.format_exc()
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise


run()

