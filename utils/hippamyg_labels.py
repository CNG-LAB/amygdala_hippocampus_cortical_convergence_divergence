#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified label helpers for hippocampus & amygdala.

Hippocampus:
    - build_hippocampus_labels(side, n_labels, subfield_order=None, order='subfield-major')
      Returns (names, ap_bins) where len(names) == n_labels and ap_bins are integers 1..K,
      with K = n_labels / len(subfield_order).

      Default subfield order is Subiculum, CA1, CA2, CA3, CA4 (Subiculum-first, per the dataset).
      Set order='ap-major' if that ordering is ever needed.

Amygdala:
    - get_amygdala_labels(side=None, style='long')
      If side is None, returns base 9 amygdala subnuclei names.
      If side is 'Left' or 'Right', prefixes each with the side.
      style='long' uses full names; style='short' gives compact labels.

Index helper:
    - region_indices(n_hipp, n_amyg=9)
      Returns dict with left/right amyg/hipp index arrays in the standard matrix layout:
        [L-Amyg(9), R-Amyg(9), L-Hipp(n_hipp), R-Hipp(n_hipp), Cortex(...)]
"""

from typing import List, Tuple, Optional
import numpy as np

SUBFIELD_ORDER_DEFAULT: Tuple[str, ...] = ("Subiculum","CA1", "CA2", "CA3", "CA4")

AMYGDALA_LONG: Tuple[str, ...] = (
    "Lateral",
    "Basal",
    "Central",
    "Medial",
    "Cortical",
    "Accessory-Basal",
    "Cortico-Amygdaloid-Transition",
    "Anterior-Amygdaloid",
    "Paralaminar",
)

AMYGDALA_SHORT: Tuple[str, ...] = (
    "Lateral",
    "Basal",
    "Central",
    "Medial",
    "Cortical",
    "Acc-Basal",
    "CAT",
    "Ant-Amyg",
    "Paralaminar",
)

# FreeSurfer-style LUT RGBA for each amygdala subnucleus (0–255)
_AMYGDALA_LUT_RGBA = {
    "Lateral":                       (72, 132, 181, 0),
    "Basal":                         (207, 63, 79, 0),
    "Central":                       (197, 60, 248, 0),
    "Medial":                        (2, 149, 2, 0),
    "Cortical":                      (221, 249, 166, 0),
    "Accessory-Basal":               (232, 146, 35, 0),
    "Cortico-Amygdaloid-Transition": (20, 60, 120, 0),
    "Anterior-Amygdaloid":           (250, 250, 0, 0),
    "Paralaminar":                   (45, 205, 165, 0),
}

# Mapping for the canonical 3-way grouping:
#   - Basolateral: Lateral, Basal, Accessory-Basal, Paralaminar
#   - Centromedial: Central, Medial
#   - Superficial: Cortical, Cortico-Amygdaloid-Transition, Anterior-Amygdaloid
#
# Note: This follows common BLA/CMA/SFA usage for the Saygin/FreeSurfer atlas.

_AMYGDALA_BROAD_MAP_LONG = {
    "Lateral":                        "Basolateral",
    "Basal":                          "Basolateral",
    "Accessory-Basal":               "Basolateral",
    "Paralaminar":                   "Basolateral",
    "Central":                        "Centromedial",
    "Medial":                         "Centromedial",
    "Cortical":                       "Superficial",
    "Cortico-Amygdaloid-Transition": "Superficial",
    "Anterior-Amygdaloid":           "Superficial",
}


def build_hippocampus_labels(
    side: str,
    n_labels: int,
    subfield_order: Optional[Tuple[str, ...]] = None,
    order: str = "subfield-major",
) -> Tuple[List[str], List[int]]:
    """
    Create hippocampus label names and their A–P bin indices.

    Args
    ----
    side : 'Left' or 'Right'
    n_labels : total labels for the hemisphere (must be multiple of #subfields)
    subfield_order : sequence of subfields (default Subiculum, CA1, CA2, CA3, CA4)
    order : 'subfield-major' (default) or 'ap-major'

    Returns
    -------
    names : list[str]  -> ["Left Sub AP1", "Left Sub AP2", ..., "Left CA1 AP1", ...]
    ap_bins : list[int] -> [1,2,...] per column; length == n_labels
    """
    if subfield_order is None:
        subfield_order = SUBFIELD_ORDER_DEFAULT
    assert side in ("Left", "Right"), "side must be 'Left' or 'Right'"
    assert n_labels > 0 and n_labels % len(subfield_order) == 0, \
        f"n_labels must be positive and divisible by {len(subfield_order)}"

    K = n_labels // len(subfield_order)  # #AP bins
    names: List[str] = []
    ap_bins: List[int] = []

    if order == "subfield-major":
        # Sub AP1..K, CA1 AP1..K, ... CA4 AP1..K
        for sf in subfield_order:
            for ap in range(1, K + 1):
                names.append(f"{side} {sf} AP{ap}")
                ap_bins.append(ap)
    elif order == "ap-major":
        # AP1: (Sub, CA1, ...), AP2: (Sub, CA1, ...), ...
        for ap in range(1, K + 1):
            for sf in subfield_order:
                names.append(f"{side} {sf} AP{ap}")
                ap_bins.append(ap)
    else:
        raise ValueError("order must be 'subfield-major' or 'ap-major'")

    return names, ap_bins


def get_amygdala_labels(side: Optional[str] = None, style: str = "long") -> List[str]:
    """
    Get amygdala subnuclei names.

    Args
    ----
    side : None -> base names; 'Left' or 'Right' -> sided names
    style: 'long' (full names) or 'short' (compact)

    Returns
    -------
    list[str]
    """
    base = AMYGDALA_LONG if style == "long" else AMYGDALA_SHORT
    if side is None:
        return list(base)
    assert side in ("Left", "Right")
    return [f"{side} {b}" for b in base]


def region_indices(n_hipp: int, n_amyg: int = 9):
    """
    Standard row layout indices for the subcortical block:
        [L-Amyg(9), R-Amyg(9), L-Hipp(n_hipp), R-Hipp(n_hipp)]
    """
    left_amyg  = np.arange(0, n_amyg)
    right_amyg = np.arange(n_amyg, 2 * n_amyg)
    left_hipp  = np.arange(2 * n_amyg, 2 * n_amyg + n_hipp)
    right_hipp = np.arange(2 * n_amyg + n_hipp, 2 * n_amyg + 2 * n_hipp)
    return {
        "Left":  {"amyg_indices": left_amyg,  "hipp_indices": left_hipp},
        "Right": {"amyg_indices": right_amyg, "hipp_indices": right_hipp},
    }




def amygdala_broad_group_indices() -> dict:
    """
    Return index arrays (0-based) for each broad group within the standard
    9-nucleus ordering used in AMYGDALA_LONG.

    Returns
    -------
    dict
        {
          "Basolateral":  np.array([...]),
          "Centromedial": np.array([...]),
          "Superficial":  np.array([...]),
        }

    Notes
    -----
    This is hard-coded to the canonical AMYGDALA_LONG ordering, which is
    also the assumption in region_indices(...).
    """
    base = list(AMYGDALA_LONG)  # length 9, canonical ordering
    group_to_idx = {}

    for idx, name in enumerate(base):
        grp = _AMYGDALA_BROAD_MAP_LONG.get(name)
        if grp is None:
            raise KeyError(f"No broad-group mapping defined for amygdala label: {name!r}")
        group_to_idx.setdefault(grp, []).append(idx)

    # Convert lists to numpy arrays for convenience in indexing
    return {k: np.asarray(v, dtype=int) for k, v in group_to_idx.items()}
    

