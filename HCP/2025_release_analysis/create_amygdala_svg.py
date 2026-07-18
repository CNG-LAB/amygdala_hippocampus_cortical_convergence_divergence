#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create SVG nucleus outlines from FreeSurfer hippoAmyg label volumes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nb
import svgwrite
from scipy.ndimage import binary_closing, binary_opening
from shapely.geometry import MultiPolygon, Polygon
from skimage import measure

# -------------------------
# Configuration
# -------------------------

RESOURCE_ROOT = Path(os.environ["HIPPAMYG_RESOURCE_ROOT"])
OUTPUT_ROOT = Path(os.environ["HIPPAMYG_OUTPUT_ROOT"])
IN_LH = str(RESOURCE_ROOT / "lh.hippoAmygLabels.mgz")
IN_RH = str(RESOURCE_ROOT / "rh.hippoAmygLabels.mgz")
OUT_DIR = str(OUTPUT_ROOT / "schematics/amygdala")

# Axis to slice: 'x' sagittal, 'y' coronal, 'z' axial
AXIS = "x"

# Slice index; None selects the slice with the largest label area.
SLICE_IDX = 72

# Reorient the label image to closest-canonical RAS coordinates.
CANONICAL = True

# Apply light binary opening and closing before contouring.
DO_OPEN_CLOSE = True

# Reflect the slice vertically in SVG coordinates.
FLIP_Y = False

# Contour/polygon filtering
MIN_AREA_PX2 = 1.0

# Simplification tolerance in pixels.
SIMPLIFY_TOL = 0

# Default SVG styling
DEFAULT_FILL = "#FFFFFF"
STROKE = "#000000"
STROKE_WIDTH = 1.2
LABEL_FONT_SIZE_PX = 12

# Amygdala LUT IDs used in the manuscript.
_AMYGDALA_LUT_IDS: Dict[str, int] = {
    "Lateral":                       7001,
    "Basal":                         7003,
    "Central":                       7005,
    "Medial":                        7006,
    "Cortical":                      7007,
    "Accessory-Basal":               7008,
    "Cortico-Amygdaloid-Transition": 7009,
    "Anterior-Amygdaloid":           7010,
    "Paralaminar":                   7015,
}

# Short nucleus names used as SVG element IDs
_SHORT_ID: Dict[str, str] = {
    "Anterior-Amygdaloid":           "AAA",
    "Accessory-Basal":               "AB",
    "Basal":                         "Ba",
    "Lateral":                       "La",
    "Central":                       "Ce",
    "Cortical":                      "Co",
    "Medial":                        "Me",
    "Cortico-Amygdaloid-Transition": "CAT",
    "Paralaminar":                   "PL",
}


def load_label_volume(path: str, canonical: bool) -> np.ndarray:
    img = nb.load(path)
    if canonical:
        img = nb.as_closest_canonical(img)
    data = np.asanyarray(img.dataobj)
    data = np.squeeze(data)  # handles 4D with singleton last dim
    if data.ndim != 3:
        raise ValueError(f"Expected 3D after squeeze; got shape={data.shape} from {path}")
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data).astype(np.int32)
    return data


def get_slice(data: np.ndarray, dim: int, idx: int) -> np.ndarray:
    sl = np.take(data, indices=idx, axis=dim)
    if sl.ndim != 2:
        raise ValueError(f"Slice not 2D (got {sl.ndim}D). data.shape={data.shape}, dim={dim}, idx={idx}")
    return sl


def pick_best_slice_for_labels(data: np.ndarray, dim: int, label_values: List[int]) -> int:
    label_set = set(int(v) for v in label_values)
    best_i = 0
    best_count = -1
    for i in range(data.shape[dim]):
        sl = get_slice(data, dim, i)
        count = int(np.count_nonzero(np.isin(sl, list(label_set))))
        if count > best_count:
            best_count = count
            best_i = i
    return best_i


def maybe_morphology(mask: np.ndarray, do_open_close: bool) -> np.ndarray:
    if not do_open_close:
        return mask
    m = binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    m = binary_closing(m, structure=np.ones((3, 3), dtype=bool))
    return m


def contours_to_polygon_xy(mask: np.ndarray) -> Optional[np.ndarray]:
    # find_contours gives (row, col) subpixel coords for boundary at level 0.5
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    if not contours:
        return None
    # Use the longest contour (usually the outer boundary)
    c = max(contours, key=lambda a: a.shape[0])
    # Convert to (x, y) = (col, row)
    xy = np.stack([c[:, 1], c[:, 0]], axis=1)
    return xy


def simplify_and_filter_polygon(xy: np.ndarray, tol: float, min_area: float) -> Optional[np.ndarray]:
    if xy.shape[0] < 3:
        return None

    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None

    poly_s = poly.simplify(tol, preserve_topology=True)
    if poly_s.is_empty:
        return None

    if isinstance(poly_s, MultiPolygon):
        poly_s = max(list(poly_s.geoms), key=lambda g: g.area)

    if poly_s.area < min_area:
        return None

    coords = np.array(poly_s.exterior.coords, dtype=float)
    return coords


def polygon_centroid(xy: np.ndarray) -> Tuple[float, float]:
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    centroid = poly.centroid
    return float(centroid.x), float(centroid.y)


def xy_to_svg_path(xy: np.ndarray, flip_y: bool, height: int) -> str:
    pts = np.array(xy, dtype=float)
    if flip_y:
        pts[:, 1] = (height - 1) - pts[:, 1]
    parts = [f"M {pts[0,0]:.2f} {pts[0,1]:.2f}"]
    for i in range(1, pts.shape[0]):
        parts.append(f"L {pts[i,0]:.2f} {pts[i,1]:.2f}")
    parts.append("Z")
    return " ".join(parts)


def export_one_hemi(
    in_path: str,
    hemi_tag: str,
    out_dir: str,
    axis: str,
    slice_idx: Optional[int],
    canonical: bool,
    do_open_close: bool,
    flip_y: bool,
) -> Tuple[Path, Path]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    data = load_label_volume(in_path, canonical=canonical)
    axis_key = axis.lower()
    if axis_key not in ("x", "y", "z"):
        raise ValueError("AXIS must be one of: 'x','y','z'")
    dim = {"x": 0, "y": 1, "z": 2}[axis_key]
    full_names = list(_AMYGDALA_LUT_IDS.keys())
    label_vals = [int(_AMYGDALA_LUT_IDS[k]) for k in full_names]
    short_ids = [_SHORT_ID[k] for k in full_names]

    if slice_idx is None:
        slice_idx = pick_best_slice_for_labels(data, dim, label_vals)
    if not 0 <= slice_idx < data.shape[dim]:
        raise ValueError(
            f"Slice {slice_idx} is outside axis {axis} with size {data.shape[dim]}"
        )

    sl = get_slice(data, dim, slice_idx)
    h, w = sl.shape

    svg_path = out_dir_p / f"amyg_{hemi_tag.upper()}_axis{axis.upper()}_slice{slice_idx}.svg"
    json_path = out_dir_p / f"amyg_{hemi_tag.upper()}_axis{axis.upper()}_slice{slice_idx}.json"

    dwg = svgwrite.Drawing(str(svg_path), size=(f"{w}px", f"{h}px"))
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="white"))

    polys_out: Dict[str, Dict[str, object]] = {}
    present_labels = set(int(v) for v in np.unique(sl) if int(v) != 0)

    for full_name, lab, sid in zip(full_names, label_vals, short_ids):
        if lab not in present_labels:
            continue

        mask = (sl == lab)
        if np.count_nonzero(mask) == 0:
            continue

        mask = maybe_morphology(mask, do_open_close=do_open_close)
        xy = contours_to_polygon_xy(mask)
        if xy is None:
            continue

        xy2 = simplify_and_filter_polygon(xy, tol=SIMPLIFY_TOL, min_area=MIN_AREA_PX2)
        if xy2 is None:
            continue

        path_d = xy_to_svg_path(xy2, flip_y=flip_y, height=h)
        cx, cy = polygon_centroid(xy2)
        if flip_y:
            cy = (h - 1) - cy

        # nucleus polygon
        dwg.add(
            dwg.path(
                d=path_d,
                id=sid,
                fill=DEFAULT_FILL,
                stroke=STROKE,
                stroke_width=STROKE_WIDTH,
            )
        )

        # Nucleus label
        dwg.add(
            dwg.text(
                sid,
                insert=(cx, cy),
                text_anchor="middle",
                alignment_baseline="middle",
                font_size=f"{LABEL_FONT_SIZE_PX}px",
                fill="#000000",
            )
        )

        polys_out[sid] = {
            "full_name": full_name,
            "label_value": int(lab),
            "slice_axis": axis,
            "slice_index": int(slice_idx),
            "points_xy": np.asarray(xy2).tolist(),
            "centroid_xy": [float(cx), float(cy)],
        }

    dwg.save()
    json_path.write_text(json.dumps(polys_out, indent=2))

    print(f"[{hemi_tag}] input: {in_path}")
    print(f"[{hemi_tag}] data shape: {data.shape} | axis={axis} (dim={dim}) | slice={slice_idx} | 2D={sl.shape}")
    print(f"[{hemi_tag}] wrote: {svg_path}")
    print(f"[{hemi_tag}] wrote: {json_path}")
    return svg_path, json_path


def main():
    if not os.path.exists(IN_LH):
        raise FileNotFoundError(f"Left label file not found: {IN_LH}")
    if not os.path.exists(IN_RH):
        raise FileNotFoundError(f"Right label file not found: {IN_RH}")

    export_one_hemi(
        in_path=IN_LH,
        hemi_tag="LH",
        out_dir=OUT_DIR,
        axis=AXIS,
        slice_idx=SLICE_IDX,
        canonical=CANONICAL,
        do_open_close=DO_OPEN_CLOSE,
        flip_y=FLIP_Y,
    )

    export_one_hemi(
        in_path=IN_RH,
        hemi_tag="RH",
        out_dir=OUT_DIR,
        axis=AXIS,
        slice_idx=SLICE_IDX,
        canonical=CANONICAL,
        do_open_close=DO_OPEN_CLOSE,
        flip_y=FLIP_Y,
    )


if __name__ == "__main__":
    main()
