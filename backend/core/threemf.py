"""Colored 3MF writer (F-23).

A 3MF file is a zip (OPC package) holding one XML model part. This writes it
directly — trimesh's 3MF export can't do per-triangle properties, and the
format is simple enough that a dependency isn't warranted. Colors use the 3MF
Materials & Properties extension: one <m:colorgroup> palette, each triangle
referencing a palette index via pid/p1. Bambu Studio maps the palette to AMS
slots on import (up to 32 colors).

Faces with class -1 (the base bottom and walls) carry no p1 and fall back to
the object-level default pindex.
"""

import re
import zipfile

CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
MATERIAL_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"

CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""

RELS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""

HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def write_3mf(path, vertices_mm, faces, face_classes, colors_hex, default_index):
    """Write a colored 3MF: mesh in millimetres, one color per face.

    face_classes: int per face — an index into colors_hex, or -1 to use the
    object default (default_index). Fails loudly on malformed input rather
    than emitting a file a slicer will misread."""
    colors_hex = list(colors_hex)
    if not colors_hex:
        raise ValueError("3MF export needs at least one palette color.")
    for c in colors_hex:
        if not HEX_COLOR_RE.match(c):
            raise ValueError(
                f"Palette color {c!r} is not #RRGGBB hex — check the color inputs."
            )
    if len(face_classes) != len(faces):
        raise ValueError(
            f"face_classes length {len(face_classes)} != face count {len(faces)}."
        )
    if not (0 <= int(default_index) < len(colors_hex)):
        raise ValueError("Default color index is outside the palette.")

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="millimeter" xml:lang="en-US" xmlns="{CORE_NS}" '
        f'xmlns:m="{MATERIAL_NS}">\n'
        " <resources>\n"
        '  <m:colorgroup id="1">\n'
    ]
    parts.extend(f'   <m:color color="{c}"/>\n' for c in colors_hex)
    parts.append("  </m:colorgroup>\n")
    parts.append(
        f'  <object id="2" type="model" pid="1" pindex="{int(default_index)}">\n'
        "   <mesh>\n    <vertices>\n"
    )
    parts.extend(
        f'     <vertex x="{x:.6g}" y="{y:.6g}" z="{z:.6g}"/>\n'
        for x, y, z in vertices_mm
    )
    parts.append("    </vertices>\n    <triangles>\n")

    n_colors = len(colors_hex)
    for (v1, v2, v3), cls in zip(faces, face_classes):
        cls = int(cls)
        if cls == -1:
            parts.append(f'     <triangle v1="{v1}" v2="{v2}" v3="{v3}"/>\n')
        elif 0 <= cls < n_colors:
            parts.append(f'     <triangle v1="{v1}" v2="{v2}" v3="{v3}" p1="{cls}"/>\n')
        else:
            raise ValueError(f"Face class {cls} is outside the palette.")

    parts.append(
        "    </triangles>\n   </mesh>\n  </object>\n </resources>\n"
        ' <build>\n  <item objectid="2"/>\n </build>\n</model>\n'
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        z.writestr("_rels/.rels", RELS_XML)
        z.writestr("3D/3dmodel.model", "".join(parts))
    return path
