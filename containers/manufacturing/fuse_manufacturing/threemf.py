from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"


def export_3mf(mesh: trimesh.Trimesh, path: str | Path, title: str = "FUSE repair") -> None:
    """Write a minimal standards-compliant, millimetre-based 3MF package."""
    path = Path(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    ET.register_namespace("", CORE_NS)
    model = ET.Element(f"{{{CORE_NS}}}model", {"unit": "millimeter", "xml:lang": "en-US"})
    metadata = ET.SubElement(model, f"{{{CORE_NS}}}metadata", {"name": "Title"})
    metadata.text = title
    resources = ET.SubElement(model, f"{{{CORE_NS}}}resources")
    object_node = ET.SubElement(
        resources,
        f"{{{CORE_NS}}}object",
        {"id": "1", "type": "model", "name": title},
    )
    mesh_node = ET.SubElement(object_node, f"{{{CORE_NS}}}mesh")
    vertices_node = ET.SubElement(mesh_node, f"{{{CORE_NS}}}vertices")
    for vertex in vertices:
        ET.SubElement(
            vertices_node,
            f"{{{CORE_NS}}}vertex",
            {"x": f"{vertex[0]:.12g}", "y": f"{vertex[1]:.12g}", "z": f"{vertex[2]:.12g}"},
        )
    triangles_node = ET.SubElement(mesh_node, f"{{{CORE_NS}}}triangles")
    for face in faces:
        ET.SubElement(
            triangles_node,
            f"{{{CORE_NS}}}triangle",
            {"v1": str(int(face[0])), "v2": str(int(face[1])), "v3": str(int(face[2]))},
        )
    build = ET.SubElement(model, f"{{{CORE_NS}}}build")
    ET.SubElement(build, f"{{{CORE_NS}}}item", {"objectid": "1"})
    model_bytes = ET.tostring(model, encoding="utf-8", xml_declaration=True)

    types = ET.Element(f"{{{CONTENT_NS}}}Types")
    ET.SubElement(
        types,
        f"{{{CONTENT_NS}}}Default",
        {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"},
    )
    ET.SubElement(
        types,
        f"{{{CONTENT_NS}}}Default",
        {"Extension": "model", "ContentType": "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"},
    )
    types_bytes = ET.tostring(types, encoding="utf-8", xml_declaration=True)

    relationships = ET.Element(f"{{{REL_NS}}}Relationships")
    ET.SubElement(
        relationships,
        f"{{{REL_NS}}}Relationship",
        {
            "Target": "/3D/3dmodel.model",
            "Id": "rel0",
            "Type": "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel",
        },
    )
    relationships_bytes = ET.tostring(relationships, encoding="utf-8", xml_declaration=True)

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", types_bytes)
        archive.writestr("_rels/.rels", relationships_bytes)
        archive.writestr("3D/3dmodel.model", model_bytes)

